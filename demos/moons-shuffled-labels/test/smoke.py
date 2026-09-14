"""Opens index.html in headless Chromium, switches widths, trains both panels, shows the test points,
checks that nothing throws, and saves screenshots to test/shots/.
Run from the demo directory:
  uv run --with playwright python test/smoke.py
"""
import glob, math, os, re, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = 'file://' + os.path.join(HERE, '..', 'index.html')
SHOTS = os.path.join(HERE, 'shots')
CHROME = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[-1]
H = 'MSUI.hooks'

# non-white pixels (every 7th) on each panel canvas
INK = """() => Array.from(document.querySelectorAll('canvas.pc')).map(cv => {
  const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data; let ink = 0;
  for (let i = 0; i < d.length; i += 4*7) if (d[i] < 200 || d[i+1] < 200 || d[i+2] < 200) ink++;
  return ink;
})"""
PERCENT = re.compile(r'\d+(\.\d)?%')


def open_page(pg, query=''):
    pg.goto(URL + query)   # a new query string reloads the page
    pg.wait_for_function('window.MSUI && MSUI.hooks', timeout=10000)
    pg.wait_for_timeout(300)
    return pg.evaluate(f'{H}.state()')


def shot(pg, name, full=True):
    pg.evaluate(f'{H}.shadeNow()')
    pg.screenshot(path=os.path.join(SHOTS, name), full_page=full)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    os.makedirs(SHOTS, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={'width': 1600, 'height': 1000})
        pg.on('pageerror', lambda e: errors.append(str(e)))
        pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)

        # default page: width 64, nothing trained, both panels drawn, English, test numbers hidden
        st = open_page(pg)
        assert st['width'] == 64 and st['steps'] == 0 and st['lang'] == 'en' and not st['showTest'], st
        ink = pg.evaluate(INK)
        assert len(ink) == 2 and min(ink) > 200, ink
        assert pg.inner_text('#topbar .brand').startswith('Same Moons, Shuffled Labels')
        assert pg.inner_text('#panelReal h3') == 'Real labels' and pg.inner_text('#panelShuf h3') == 'Shuffled labels'
        assert pg.inner_text('#panelReal b.te') == '–' and PERCENT.fullmatch(pg.inner_text('#panelShuf b.tr'))
        preset = pg.evaluate('MS.PRESET')
        shot(pg, 'open.png')

        # width chip: new network at that width, step counter back to 0
        pg.click('#segWidth button[data-v="16"]')
        st = pg.evaluate(f'{H}.state()')
        assert st['width'] == 16 and st['steps'] == 0, st
        pg.evaluate(f'{H}.draw()')   # the page redraws on the next frame; draw now before reading classes
        assert 'on' in (pg.get_attribute('#segWidth button[data-v="16"]', 'class') or '')
        assert 'on' not in (pg.get_attribute('#segWidth button[data-v="64"]', 'class') or '')
        loss0 = st['real']['loss']

        # training a few steps: both losses are finite and the counter follows
        st = pg.evaluate(f'{H}.train(50)')
        assert st['steps'] == 50 and pg.inner_text('#sSteps') == '50', st
        for k in ('real', 'shuf'):
            assert finite(st[k]['loss']) and finite(st[k]['trainAcc']), st

        # test points: both panels show a test accuracy
        pg.evaluate(f'{H}.setShowTest(true)')
        pg.evaluate(f'{H}.draw()')
        assert pg.evaluate(f'{H}.state()')['showTest']
        assert 'on' in (pg.get_attribute('#bTest', 'class') or '')
        for panel in ('#panelReal', '#panelShuf'):
            te = pg.inner_text(panel + ' b.te')
            assert PERCENT.fullmatch(te), (panel, te)

        # Reset draws new starting knobs
        pg.evaluate(f'{H}.reset()')
        st = pg.evaluate(f'{H}.state()')
        assert st['steps'] == 0 and st['width'] == 16 and st['real']['loss'] != loss0, (loss0, st)

        # Space starts and pauses the frame loop; with the focus on Reset it must not also press Reset
        pg.focus('#bReset')
        pg.keyboard.press('Space')
        pg.wait_for_timeout(700)
        st = pg.evaluate(f'{H}.state()')
        assert st['training'] and st['width'] == 16 and pg.inner_text('#bTrain') == 'Pause', st
        pg.keyboard.press('Space')
        pg.wait_for_timeout(100)
        st = pg.evaluate(f'{H}.state()')
        assert not st['training'] and st['steps'] > 0 and pg.inner_text('#bTrain') == 'Train', st

        # the frame loop pauses by itself once both panels classify every training point correctly
        open_page(pg)
        pg.select_option('#speed', '20')
        t0 = time.time()
        pg.click('#bTrain')
        pg.wait_for_function(f'{H}.state().training === false', timeout=180000, polling=200)
        st = pg.evaluate(f'{H}.state()')
        assert st['real']['trainAcc'] == 1 and st['shuf']['trainAcc'] == 1, st
        assert pg.inner_text('#note') == 'Paused: both networks classify every training point correctly.', pg.inner_text('#note')
        print(f'frame loop at width 64, 20 steps per frame: paused at step {st["steps"]} after {time.time() - t0:.1f} s')

        # capture: width 64 trained with train() until both panels are at 100%
        st = open_page(pg)
        while not (st['real']['trainAcc'] == 1 and st['shuf']['trainAcc'] == 1):
            assert st['steps'] < preset['maxSteps'], st
            st = pg.evaluate(f'{H}.train(10)')
        shot(pg, 'fit-64.png')
        pg.evaluate(f'{H}.setShowTest(true)')
        st = pg.evaluate(f'{H}.state()')
        shot(pg, 'fit-64-test.png')
        print('width 64 fit at step', st['steps'], '| test accuracy real', st['real']['testAcc'], 'shuffled', st['shuf']['testAcc'])
        assert st['real']['testAcc'] >= 0.9 and st['shuf']['testAcc'] <= 0.7, st
        pg.set_viewport_size({'width': 1280, 'height': 860})
        pg.wait_for_timeout(200)
        pg.evaluate(f'{H}.shadeNow()')
        pg.locator('#panels').screenshot(path=os.path.join(SHOTS, 'fit-64-panels-1280.png'))
        pg.locator('#lossWrap').screenshot(path=os.path.join(SHOTS, 'fit-64-loss-1280.png'))
        pg.set_viewport_size({'width': 1600, 'height': 1000})

        # URL options: width and test points
        st = open_page(pg, '?width=4&test=1')
        assert st['width'] == 4 and st['showTest'], st
        assert PERCENT.fullmatch(pg.inner_text('#panelReal b.te'))
        st = pg.evaluate(f'{H}.train({preset["steps"]})')
        print('width 4 after', st['steps'], 'steps: training accuracy real', st['real']['trainAcc'], 'shuffled', st['shuf']['trainAcc'])
        assert st['shuf']['trainAcc'] < 1, st
        shot(pg, 'width-4.png')

        # the frame loop stops at PRESET.maxSteps
        pg.select_option('#speed', '20')
        pg.evaluate(f'{H}.startTraining()')
        pg.wait_for_function(f'{H}.state().training === false', timeout=180000, polling=200)
        st = pg.evaluate(f'{H}.state()')
        assert st['steps'] == preset['maxSteps'], st
        assert pg.inner_text('#note') == 'Paused after {:,} steps.'.format(preset['maxSteps']), pg.inner_text('#note')

        # train=1 starts training on load
        st = open_page(pg, '?width=4&train=1')
        pg.wait_for_timeout(500)
        st = pg.evaluate(f'{H}.state()')
        assert st['training'] and st['steps'] > 0, st
        pg.evaluate(f'{H}.stopTraining()')

        # width 256 builds, trains, and shades; print how fast it runs
        st = open_page(pg, '?width=256')
        t0 = time.time()
        st = pg.evaluate(f'{H}.train(10)')
        t1 = time.time()
        pg.evaluate(f'{H}.shadeNow()')
        t2 = time.time()
        assert st['width'] == 256 and st['steps'] == 10, st
        assert finite(st['real']['loss']) and finite(st['shuf']['loss']), st
        pg.evaluate(f'{H}.startTraining()')
        pg.wait_for_timeout(3000)
        pg.evaluate(f'{H}.stopTraining()')
        n = pg.evaluate(f'{H}.state()')['steps'] - 10
        print(f'width 256: train(10) {t1 - t0:.2f} s, shadeNow {t2 - t1:.2f} s, frame loop {n} steps in 3 s')

        # Korean
        st = open_page(pg, '?lang=ko')
        assert st['lang'] == 'ko' and pg.evaluate('document.documentElement.lang') == 'ko', st
        assert pg.inner_text('#topbar .brand').startswith('같은 초승달, 섞은 라벨'), pg.inner_text('#topbar .brand')
        assert pg.inner_text('#panelShuf h3') == '섞은 라벨'
        assert pg.inner_text('#bTrain') == '학습' and '학습률' in pg.inner_text('#fine')
        pg.evaluate(f'{H}.train(300)')
        pg.evaluate(f'{H}.setShowTest(true)')
        shot(pg, 'ko-300.png')

        # narrow window: the panels stack
        pg.set_viewport_size({'width': 1100, 'height': 900})
        pg.wait_for_timeout(200)
        pg.evaluate(f'{H}.draw()')
        r1 = pg.locator('#panelReal').bounding_box()
        r2 = pg.locator('#panelShuf').bounding_box()
        assert r2['y'] >= r1['y'] + r1['height'] - 1, (r1, r2)
        assert pg.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'horizontal scroll'
        shot(pg, 'narrow.png')
        b.close()

    if errors:
        print('ERRORS:')
        for e in errors:
            print('  ', e)
        sys.exit(1)
    print('smoke ok')


if __name__ == '__main__':
    main()
