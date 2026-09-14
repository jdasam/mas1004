"""Opens index.html in headless Chromium, switches datasets, activations and layers,
plays and trains, checks that nothing throws, and saves screenshots to test/shots/.
Run from the demo directory:
  uv run --with playwright python test/smoke.py
"""
import glob, os, sys
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = 'file://' + os.path.join(HERE, '..', 'index.html')
SHOTS = os.path.join(HERE, 'shots')
CHROME = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[-1]


def main():
    os.makedirs(SHOTS, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={'width': 1600, 'height': 1000})
        pg.on('pageerror', lambda e: errors.append(str(e)))
        pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        pg.goto(URL)
        pg.wait_for_timeout(300)
        H = 'LBUI.hooks'

        st = pg.evaluate(f'{H}.state()')
        assert st['data'] == 'moons' and st['K'] > 0 and st['lang'] == 'en', st
        assert pg.locator('#topbar .brand').inner_text().startswith('Layer by Layer in 2D')
        pg.screenshot(path=os.path.join(SHOTS, '00-open.png'))

        # every dataset x activation builds, draws, and trains a little
        for data in ['blobs', 'moons', 'xor', 'spirals']:
            pg.evaluate(f'{H}.setData("{data}")')
            for act in ['tanh', 'leaky', 'relu']:
                pg.evaluate(f'{H}.setAct("{act}")')
                r = pg.evaluate(f'{H}.train(20)')
                assert r['loss'] == r['loss'], f'loss is NaN for {data} {act}'
                st = pg.evaluate(f'{H}.state()')
                assert st['act'] == act and st['steps'] == 20, st
                checks = pg.evaluate('LBUI.selfCheck()')
                for name, ok in checks.items():
                    if not ok:
                        errors.append(f'selfCheck {name} failed for {data} {act}')

        # layers: down to 0, up to 8, buttons disable at the ends
        pg.evaluate(f'{H}.setData("spirals")')
        for _ in range(6):
            pg.evaluate(f'{H}.removeLayer()')
        st = pg.evaluate(f'{H}.state()')
        assert st['layers'] == 0 and st['K'] == 1, st
        assert pg.locator('#layMinus').is_disabled()
        pg.evaluate(f'{H}.draw()')
        checks = pg.evaluate('LBUI.selfCheck()')
        assert all(checks.values()), checks
        assert pg.locator('#strip .thumb').count() == 2
        pg.screenshot(path=os.path.join(SHOTS, 'spirals-0-layers.png'))
        for _ in range(9):
            pg.evaluate(f'{H}.addLayer()')
        st = pg.evaluate(f'{H}.state()')
        assert st['layers'] == 8 and pg.locator('#layPlus').is_disabled(), st
        assert pg.locator('#strip .thumb').count() == 10
        pg.evaluate(f'{H}.setData("spirals")')

        # train the spiral preset and capture every stage at rest
        r = pg.evaluate(f'{H}.train(3000)')
        print('spirals preset after 3000 steps:', r)
        st = pg.evaluate(f'{H}.state()')
        K = st['K']
        for k in range(K + 1):
            pg.evaluate(f'{H}.setPos({k})')
            pg.screenshot(path=os.path.join(SHOTS, f'spirals-{k:02d}.png'))
        # mid-step frames for the first layer: rotate, stretch, rotate, shift, activation
        for k in range(0, 7):
            pg.evaluate(f'{H}.setPos({k} + 0.55)')
            pg.screenshot(path=os.path.join(SHOTS, f'spirals-mid-{k:02d}.png'))

        # playing one step ends exactly at the next boundary
        pg.evaluate(f'{H}.setPos(0)')
        pg.evaluate(f'{H}.playStep()')
        pg.wait_for_function(f'{H}.state().playing === false', timeout=10000)
        assert abs(pg.evaluate(f'{H}.state()')['pos'] - 1) < 1e-9

        # live training keeps the page responsive
        pg.evaluate(f'{H}.setData("moons")')
        pg.evaluate(f'{H}.startTraining()')
        pg.wait_for_timeout(1500)
        pg.evaluate(f'{H}.stopTraining()')
        st = pg.evaluate(f'{H}.state()')
        assert st['steps'] > 50, st
        pg.screenshot(path=os.path.join(SHOTS, 'moons-trained.png'))

        # new points: a click on the canvas adds one at the input, not in a hidden space
        pg.goto(URL)
        pg.wait_for_timeout(300)
        pg.evaluate(f'{H}.setPos(0)')
        box = pg.locator('#view').bounding_box()
        cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        pg.mouse.click(cx, cy)
        assert pg.evaluate(f'{H}.state()')['newPoints'] == 1
        pg.locator('#strip .thumb').nth(1).click()   # after layer 1
        pg.evaluate(f'{H}.draw()')
        pg.mouse.click(cx, cy)
        assert pg.evaluate(f'{H}.state()')['newPoints'] == 1
        pg.evaluate(f'{H}.clearPoints()')
        assert pg.evaluate(f'{H}.state()')['newPoints'] == 0

        # test points: the toggle, their accuracy, and a capture after training with one new point
        pg.evaluate(f'{H}.setShowTest(true)')
        st = pg.evaluate(f'{H}.state()')
        assert st['showTest'] and 0 <= st['testAcc'] <= 1, st
        assert 'on' in (pg.locator('#controls #bShowTest').get_attribute('class') or '').split()
        pg.evaluate(f'{H}.train(300)')
        pg.evaluate(f'{H}.setPos(0)')
        pg.mouse.click(cx, cy)
        st = pg.evaluate(f'{H}.state()')
        assert st['newPoints'] == 1 and st['steps'] == 300, st
        pg.evaluate(f'{H}.draw()')
        pg.screenshot(path=os.path.join(SHOTS, 'moons-test-points.png'))
        pg.goto(URL + '?test=1')
        pg.wait_for_timeout(300)
        assert pg.evaluate(f'{H}.state()')['showTest']

        # narrow window
        pg.set_viewport_size({'width': 1100, 'height': 900})
        pg.wait_for_timeout(200)
        pg.evaluate(f'{H}.draw()')
        pg.screenshot(path=os.path.join(SHOTS, 'narrow.png'), full_page=True)

        # URL options
        pg.goto(URL + '?data=xor&act=leaky&layers=3&color=x')   # a hash-only change would not reload the page
        pg.wait_for_timeout(300)
        st = pg.evaluate(f'{H}.state()')
        assert (st['data'], st['act'], st['layers'], st['colorBy']) == ('xor', 'leaky', 3, 'x'), st

        # Korean UI with the first scene of week 2: two blobs, no layers
        pg.goto(URL + '?lang=ko&data=blobs&layers=0')
        pg.wait_for_timeout(300)
        st = pg.evaluate(f'{H}.state()')
        assert (st['data'], st['layers'], st['lang']) == ('blobs', 0, 'ko'), st
        assert pg.locator('#topbar .brand').inner_text().startswith('층별 2차원 변환')
        assert pg.locator('#segData button').first.inner_text() == '덩어리 둘'
        pg.evaluate(f'{H}.draw()')
        pg.screenshot(path=os.path.join(SHOTS, 'ko-blobs-0.png'))

        # tabs: the default page shows Train, as before; Gradient shows only its own pane
        pg.set_viewport_size({'width': 1600, 'height': 1000})
        pg.goto(URL)
        pg.wait_for_timeout(300)
        assert pg.evaluate(f'{H}.state()')['tab'] == 'train'
        assert pg.locator('#paneTrain').is_visible() and pg.locator('#paneKnobs').is_hidden() and pg.locator('#paneGradient').is_hidden()
        pg.locator('#tabs button[data-tab=gradient]').click()
        pg.evaluate(f'{H}.draw()')
        assert pg.evaluate(f'{H}.state()')['tab'] == 'gradient'
        assert pg.locator('#paneGradient').is_visible()
        assert pg.locator('#paneKnobs').is_hidden() and pg.locator('#paneTrain').is_hidden()

        # update rule: Adam takes the preset's Adam learning rate
        pg.locator('#tabs button[data-tab=train]').click()
        pg.locator('#segRule button[data-v=adam]').click()
        st = pg.evaluate(f'{H}.state()')
        assert st['rule'] == 'adam' and st['lr'] == pg.evaluate('LB.PRESETS.moons.lr.adam'), st
        pg.evaluate(f'{H}.draw()')
        pg.screenshot(path=os.path.join(SHOTS, 'tab-train.png'), full_page=True)

        # gradient tab in Korean: nudge every knob, take a step, backpropagation, nudge one knob
        pg.goto(URL + '?data=moons&layers=1&tab=gradient&opt=gd&lang=ko')
        pg.wait_for_timeout(300)
        st = pg.evaluate(f'{H}.state()')
        assert (st['tab'], st['rule'], st['knobCount']) == ('gradient', 'gd', 9), st
        assert pg.locator('#bStep').is_disabled()
        g = pg.evaluate(f'{H}.measureAll()')
        assert g['evals'] == 10 and g['stepEnabled'], g
        r = pg.evaluate(f'{H}.takeStep()')
        assert r['after'] < r['before'], r
        g = pg.evaluate(f'{H}.gradState()')
        assert not g['stepEnabled'] and pg.locator('#bStep').is_disabled(), g
        slopes = pg.evaluate("() => [...document.querySelectorAll('#gTable tbody tr')].map(tr => tr.children[2].textContent + tr.children[3].textContent).join('')")
        assert pg.locator('#gTable tbody tr').count() == 9 and slopes == '', slopes
        g = pg.evaluate(f'{H}.measureBackprop()')
        assert g['bp'] == 1 and g['stepEnabled'], g
        pg.evaluate(f'{H}.nudge()')
        g = pg.evaluate(f'{H}.gradState()')
        assert g['nudge'] and abs(g['nudge']['slope'] - (g['nudge']['after'] - g['nudge']['before']) / g['delta']) < 1e-12, g
        pg.evaluate(f'{H}.measureAll()')
        pg.screenshot(path=os.path.join(SHOTS, 'tab-gradient.png'), full_page=True)

        # knobs tab: a slider moves the loss and the picture; one more layer adds 6 sliders
        pg.goto(URL + '?data=blobs&layers=0&tab=knobs')
        pg.wait_for_timeout(300)
        assert pg.locator('#knobRows input[type=range]').count() == 3
        loss0, sig0 = pg.evaluate(f'{H}.state()')['loss'], pg.evaluate(f'{H}.viewSignature()')
        pg.evaluate("() => { const r = document.querySelector('#knobRows input[type=range]'); r.value = 1.5; r.dispatchEvent(new Event('input', { bubbles: true })); }")
        loss1, sig1 = pg.evaluate(f'{H}.state()')['loss'], pg.evaluate(f'{H}.viewSignature()')
        assert loss1 != loss0 and sig1 != sig0, (loss0, loss1, sig0, sig1)
        pg.evaluate(f'{H}.setLayers(1)')
        assert pg.locator('#knobRows input[type=range]').count() == 9
        pg.screenshot(path=os.path.join(SHOTS, 'tab-knobs.png'), full_page=True)

        # two blobs cut by the wrong line v = (0, 1), c = 0: halos and outlines on the misclassified points
        pg.goto(URL + '?data=blobs&layers=0')
        pg.wait_for_timeout(300)
        for k, v in [(0, 0), (1, 1), (2, 0)]:
            pg.evaluate(f'{H}.setKnob(-1, {k}, {v})')
        st = pg.evaluate(f'{H}.state()')
        assert st['acc'] < 0.9, st
        pg.screenshot(path=os.path.join(SHOTS, 'blobs-wrong-halo.png'))
        b.close()

    if errors:
        print('ERRORS:')
        for e in errors:
            print('  ', e)
        sys.exit(1)
    print('smoke ok')


if __name__ == '__main__':
    main()
