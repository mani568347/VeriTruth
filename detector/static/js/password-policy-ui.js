/* VeriTruth — frontend password policy UX (backend validator stays authoritative).
   Renders a collapsible progress panel with requirement chips instead of a
   static always-visible checklist. */
(function () {
    'use strict';

    var SPECIAL = '!@#$%^&*()_+-=';

    var RULES = [
        { key: 'length',  label: '8+ characters',    full: 'At least 8 characters', test: function (p) { return p.length >= 8; } },
        { key: 'upper',   label: 'Uppercase',        full: 'One uppercase letter',  test: function (p) { return /[A-Z]/.test(p); } },
        { key: 'lower',   label: 'Lowercase',        full: 'One lowercase letter',  test: function (p) { return /[a-z]/.test(p); } },
        { key: 'number',  label: 'Number',           full: 'One number',            test: function (p) { return /[0-9]/.test(p); } },
        { key: 'special', label: 'Special char',     full: 'One special character', test: function (p) { return p.split('').some(function (c) { return SPECIAL.indexOf(c) !== -1; }); } }
    ];

    function evaluate(pw) {
        var result = {};
        RULES.forEach(function (r) { result[r.key] = r.test(pw); });
        return result;
    }

    function okCount(result) {
        return RULES.reduce(function (n, r) { return n + (result[r.key] ? 1 : 0); }, 0);
    }

    function allOk(result) {
        return okCount(result) === RULES.length;
    }

    function toast(msg, type) {
        if (typeof window.vtToast === 'function') {
            window.vtToast(msg, type || 'error');
        }
    }

    function buildPanel(container) {
        container.textContent = '';

        var box = document.createElement('div');
        box.className = 'pwr';

        var panel = document.createElement('div');
        panel.className = 'pwr__panel';

        var head = document.createElement('div');
        head.className = 'pwr__head';

        var count = document.createElement('span');
        count.className = 'pwr__count';
        count.setAttribute('data-pwr-count', '');
        count.setAttribute('aria-live', 'polite');
        count.textContent = '0 of ' + RULES.length + ' met';

        var track = document.createElement('span');
        track.className = 'pwr__track';
        var fill = document.createElement('span');
        fill.className = 'pwr__fill';
        fill.setAttribute('data-pwr-fill', '');
        track.appendChild(fill);

        head.appendChild(count);
        head.appendChild(track);

        var chips = document.createElement('ul');
        chips.className = 'pwr__chips';
        RULES.forEach(function (r) {
            var li = document.createElement('li');
            li.className = 'pwr__chip';
            li.setAttribute('data-rule', r.key);
            li.title = r.full;
            var dot = document.createElement('span');
            dot.className = 'pwr__dot';
            dot.setAttribute('aria-hidden', 'true');
            var text = document.createElement('span');
            text.textContent = r.label;
            li.appendChild(dot);
            li.appendChild(text);
            chips.appendChild(li);
        });

        var done = document.createElement('div');
        done.className = 'pwr__done';
        done.innerHTML =
            '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<polyline points="20 6 9 17 4 12"></polyline></svg>';
        var doneText = document.createElement('span');
        doneText.textContent = 'All requirements met — strong password';
        done.appendChild(doneText);

        panel.appendChild(head);
        panel.appendChild(chips);
        panel.appendChild(done);
        box.appendChild(panel);
        container.appendChild(box);
        return box;
    }

    function paintPanel(box, result, touched) {
        var met = okCount(result);
        var done = met === RULES.length;

        var count = box.querySelector('[data-pwr-count]');
        if (count) { count.textContent = done ? 'Ready' : met + ' of ' + RULES.length + ' met'; }

        var fill = box.querySelector('[data-pwr-fill]');
        if (fill) { fill.style.width = (met / RULES.length) * 100 + '%'; }

        RULES.forEach(function (r) {
            var chip = box.querySelector('[data-rule="' + r.key + '"]');
            if (!chip) { return; }
            var ok = result[r.key];
            chip.classList.toggle('is-ok', ok);
            chip.classList.toggle('is-bad', touched && !ok);
            var dot = chip.querySelector('.pwr__dot');
            if (dot) { dot.textContent = ok ? '✓' : ''; }
        });

        box.classList.toggle('is-done', done);
    }

    function init() {
        document.querySelectorAll('form[data-password-policy]').forEach(function (form) {
            var pw = form.querySelector('[data-pw]');
            if (!pw) { return; }
            var confirm = form.querySelector('[data-pw-confirm]');
            var reqBox = form.querySelector('[data-pw-requirements]');
            var errBox = form.querySelector('[data-pw-error]');
            var touched = false;
            var box = null;

            if (reqBox) {
                box = buildPanel(reqBox);
                paintPanel(box, evaluate(pw.value), false);
            }

            function setOpen(open) {
                if (box) { box.classList.toggle('is-open', open); }
            }

            function setError(msg) {
                if (!errBox) { return; }
                errBox.textContent = msg || '';
                errBox.style.display = msg ? 'flex' : 'none';
            }

            function refresh() {
                var result = evaluate(pw.value);
                if (box) { paintPanel(box, result, touched); }
                if (confirm && pw.value && confirm.value && pw.value !== confirm.value) {
                    setError('Passwords do not match.');
                } else if (touched && !allOk(result)) {
                    setError('Please meet all password requirements above.');
                } else {
                    setError('');
                }
            }

            pw.addEventListener('focus', function () { setOpen(true); });
            pw.addEventListener('blur', function () {
                window.setTimeout(function () {
                    var active = document.activeElement;
                    var inWrap = active && active.closest && active.closest('.pw-field-wrap');
                    if (active !== pw && !inWrap) { setOpen(false); }
                }, 0);
            });
            pw.addEventListener('input', function () { touched = true; setOpen(true); refresh(); });
            if (confirm) { confirm.addEventListener('input', refresh); }

            // Show/hide password toggles
            form.querySelectorAll('[data-pw-toggle]').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var target = form.querySelector(btn.getAttribute('data-pw-toggle'));
                    if (!target) { return; }
                    var showing = target.type === 'text';
                    target.type = showing ? 'password' : 'text';
                    btn.setAttribute('aria-pressed', String(!showing));
                    btn.classList.toggle('is-on', !showing);
                });
            });

            form.addEventListener('submit', function (e) {
                touched = true;
                var result = evaluate(pw.value);
                if (!allOk(result)) {
                    e.preventDefault();
                    refresh();
                    setOpen(true);
                    toast('Password does not meet the requirements.', 'error');
                    pw.focus();
                    return;
                }
                if (confirm && pw.value !== confirm.value) {
                    e.preventDefault();
                    setError('Passwords do not match.');
                    toast('Passwords do not match.', 'error');
                    confirm.focus();
                }
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}());
