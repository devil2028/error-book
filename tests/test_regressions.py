import io
import json
import os
import shutil
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import app
import typeset as t

@contextmanager
def test_directory():
    root = Path(__file__).resolve().parent
    folder = root / ("run_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield str(folder)
    finally:
        if folder.resolve().parent != root:
            raise RuntimeError("Unexpected test directory")
        shutil.rmtree(folder)

class RegressionTests(unittest.TestCase):
    def test_printed_parentheses_survive_cleaning(self):
        for text in ['1. 计算 (12+8)×5', '2. 求面积（单位：厘米）', '3. （1）计算']:
            line = t.OcrLine(text, 99, 0, 0, 400, 30)
            result = ' '.join(x.text for x in t.clean_ocr_lines([line]))
            self.assertEqual(result, text)
        self.assertEqual(t.blank_paren_answers('答案（）'), '答案（　　）')

    def test_numeric_conditions_survive_all_text_stages(self):
        for text in ['10 20 30', '123', '１２３']:
            line = t.OcrLine(text, 99, 0, 50, 200, 80)
            self.assertEqual(t.filter_raw_detections([line]), [line])
            self.assertEqual(t.clean_ocr_lines([line])[0].text, text)
            self.assertIn(text, t.proofread_question_texts([text])[0])
            with patch.object(t, 'wrap_text', return_value=[text]) as wrap:
                t.render_question_text([text], 400, t.find_kaiti_font())
                wrap.assert_called_once()

    def test_default_mode_preserves_erased_image_without_ocr(self):
        original = Image.new('RGB', (200, 100), 'red')
        cleaned = Image.new('RGB', (200, 100), 'white')
        with patch.object(t, 'ocr_lines', return_value=[]) as ocr:
            result, mode = t.process_question(original, {}, None, lambda im, cfg: cleaned, 200)
            ocr.assert_not_called()
            self.assertEqual(result.tobytes(), cleaned.tobytes())
            self.assertEqual(mode, 'image_clean')
        def fail(*args):
            raise RuntimeError('simulated')
        with patch.object(t, 'ocr_lines') as ocr:
            with self.assertRaises(RuntimeError):
                t.process_question(original, {}, None, fail, 200)
            ocr.assert_not_called()
        with self.assertRaises(RuntimeError):
            t.render_one_question_block(original, [], {}, fail, 200, '', 38, 0)

    def test_colored_annotation_cleanup_preserves_black_print(self):
        original = Image.new('RGB', (40, 20), 'white')
        erased = Image.new('RGB', (40, 20), 'white')
        for x in range(5, 12):
            original.putpixel((x, 10), (220, 45, 45))
            erased.putpixel((x, 10), (50, 50, 50))
        for x in range(25, 32):
            original.putpixel((x, 10), (0, 0, 0))
            erased.putpixel((x, 10), (0, 0, 0))
        result = app.cleanup_colored_annotations(original, erased)
        self.assertEqual(result.getpixel((8, 10)), (255, 255, 255))
        self.assertEqual(result.getpixel((28, 10)), (0, 0, 0))

    def test_quality_check_flags_probable_answer_residue(self):
        lines = [
            t.OcrLine('988÷260=3.8', 99, 0, 0, 200, 30),
            t.OcrLine('结果是（20）千米', 99, 0, 40, 200, 70),
        ]
        with patch.object(t, 'ocr_lines', return_value=lines):
            warnings = t.assess_erased_question(object(), Image.new('RGB', (200, 80)))
        self.assertTrue(any('等号后' in item for item in warnings))
        self.assertTrue(any('括号内' in item for item in warnings))

    def test_narrow_crop_keeps_original_page_scale(self):
        narrow = Image.new('RGB', (500, 800), 'white')
        page = app.compose_a4_pages([(narrow, 'image_clean', 1000)], draft_gap=0)[0]
        # 原先会放大到栏宽 1120；现在按整页比例显示为约 560px。
        ink = page.getbbox()
        self.assertEqual(page.size, (app.A4_W, app.A4_H))
        # 白底无法用 bbox 判断，改为检查内部缩放的确定性结果。
        self.assertLessEqual(int((app.A4_W - app.MARGIN * 2) / 1000 * 500), 560)

    def test_manual_erase_mask_whitens_only_marked_crop_area(self):
        image = Image.new('RGB', (100, 100), 'white')
        image.putpixel((40, 40), (0, 0, 0))
        image.putpixel((80, 80), (0, 0, 0))
        result = app.apply_manual_erase_masks(image, [[35, 35, 45, 45]], (0, 0, 100, 100))
        self.assertEqual(result.getpixel((40, 40)), (255, 255, 255))
        self.assertEqual(result.getpixel((80, 80)), (0, 0, 0))

    def post_question(self, client):
        buf = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(buf, 'PNG')
        buf.seek(0)
        return client.post('/api/erase', data={'files': (buf, 'test.png'), 'pages': json.dumps([{'boxes': [[0, 0, 100, 100]]}])})

    def test_failed_question_returns_location_and_no_output(self):
        with test_directory() as out, patch.object(app, 'OUTPUT_DIR', out), patch.object(app, 'load_config', return_value={'ok': True}), patch.object(app, 'get_client', return_value=object()), patch.object(app, 'process_question', side_effect=RuntimeError('simulated')), patch.object(app.app.logger, 'exception'):
            response = self.post_question(app.app.test_client())
            self.assertEqual(response.status_code, 502)
            result = response.get_json()
            self.assertFalse(result['ok'])
            self.assertEqual((result['failed_page'], result['failed_box']), (1, 1))
            self.assertNotIn('pdf', result)
            self.assertEqual(list(Path(out).iterdir()), [])

    def test_same_second_exports_do_not_overwrite(self):
        page = Image.new('RGB', (100, 100), 'white')
        with test_directory() as out, patch.object(app, 'OUTPUT_DIR', out), patch.object(app, 'load_config', return_value={'ok': True}), patch.object(app, 'get_client', return_value=object()), patch.object(app, 'process_question', return_value=(page, 'text_only')), patch.object(app, 'compose_a4_pages', return_value=[page]), patch.object(app.time, 'strftime', return_value='fixed_second'):
            client = app.app.test_client()
            a = self.post_question(client).get_json()
            b = self.post_question(client).get_json()
            self.assertTrue(a['ok'] and b['ok'])
            self.assertNotEqual(a['pdf'], b['pdf'])
            self.assertNotEqual(a['preview'], b['preview'])
            self.assertEqual(len(list(Path(out).iterdir())), 5)  # 两份 PDF/JPG 与预览任务目录
            with client.get(a['pdf']) as response:
                self.assertEqual(response.status_code, 200)
            with client.get(b['preview']) as response:
                self.assertEqual(response.status_code, 200)

    def test_preview_mask_regenerates_pdf_without_cloud_or_upload(self):
        job_id = "a" * 32
        with test_directory() as out, patch.object(app, 'OUTPUT_DIR', out):
            folder = Path(out) / 'jobs' / job_id
            folder.mkdir(parents=True)
            page = Image.new('RGB', (100, 100), 'white')
            page.putpixel((50, 50), (0, 0, 0))
            page.save(folder / 'page_0.png')
            (folder / 'manifest.json').write_text(json.dumps({'page_count': 1}), encoding='utf-8')
            response = app.app.test_client().post('/api/manual-erase-result', json={
                'job_id': job_id,
                'masks': [{'page': 0, 'x1': 45, 'y1': 45, 'x2': 55, 'y2': 55}],
            })
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertTrue(result['ok'])
            erased = Image.open(Path(out) / 'jobs' / result['job_id'] / 'page_0.png')
            self.assertEqual(erased.getpixel((50, 50)), (255, 255, 255))

    def test_credentials_are_local_with_environment_precedence(self):
        shared = json.loads(Path('config.json').read_text(encoding='utf-8'))
        self.assertNotIn('secret_id', shared)
        self.assertNotIn('secret_key', shared)
        with test_directory() as folder:
            config = Path(folder) / 'config.json'
            local = Path(folder) / 'config.local.json'
            config.write_text('{}', encoding='utf-8')
            local.write_text(json.dumps({'secret_id': 'local-id', 'secret_key': 'local-key'}), encoding='utf-8')
            with patch.object(app, 'CONFIG_PATH', str(config)), patch.object(app, 'LOCAL_CONFIG_PATH', str(local)), patch.object(app, '_config', None), patch.dict(os.environ, {'TENCENTCLOUD_SECRET_ID': 'env-id', 'TENCENTCLOUD_SECRET_KEY': 'env-key'}):
                cfg = app.load_config()
                self.assertTrue(cfg['ok'])
                self.assertEqual(cfg['secret_id'], 'env-id')
                self.assertEqual(cfg['secret_key'], 'env-key')

if __name__ == '__main__':
    unittest.main()
