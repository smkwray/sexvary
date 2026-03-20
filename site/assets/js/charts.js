/* sexvary charts – ECharts wrapper.
 * Provides: charts.forest, charts.hbar, charts.robustness, charts.ageProfile
 * All charts auto-resize and respect light/dark theme. */
var charts = (function () {
  'use strict';

  var instances = [];

  function isDark() {
    return document.documentElement.dataset.theme === 'dark';
  }

  function palette() {
    var d = isDark();
    return {
      bg: 'transparent',
      text: d ? '#e2e8f0' : '#2d3748',
      muted: d ? '#718096' : '#a0aec0',
      border: d ? '#2d3748' : '#e2e8f0',
      pos: d ? '#68d391' : '#38a169',
      neg: d ? '#fc8181' : '#e53e3e',
      accent: d ? '#63b3ed' : '#3182ce',
      bar1: d ? '#63b3ed' : '#3182ce',
      bar2: d ? '#f6ad55' : '#dd6b20',
      cardBg: d ? '#1e293b' : '#ffffff',
    };
  }

  function init(el, opts) {
    var chart = echarts.init(el, null, { renderer: 'svg' });
    chart.setOption(opts);
    instances.push({ el: el, chart: chart, optsFn: null });
    return chart;
  }

  function initDynamic(el, optsFn) {
    var chart = echarts.init(el, null, { renderer: 'svg' });
    chart.setOption(optsFn());
    instances.push({ el: el, chart: chart, optsFn: optsFn });
    return chart;
  }

  function refreshAll() {
    instances.forEach(function (item) {
      if (item.optsFn) {
        item.chart.dispose();
        item.chart = echarts.init(item.el, null, { renderer: 'svg' });
        item.chart.setOption(item.optsFn());
      }
    });
  }

  function resizeAll() {
    instances.forEach(function (item) { item.chart.resize(); });
  }

  window.addEventListener('themechange', refreshAll);
  var rt;
  window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(resizeAll, 150); });

  /* ── tooltip formatter helper ── */
  function vrTip(params) {
    var p = Array.isArray(params) ? params[0] : params;
    var datum = p.data || {};
    var vr = typeof p.value === 'number' ? p.value : (Array.isArray(p.value) ? p.value[0] : p.value);
    var dir = vr >= 1 ? 'Male > Female' : 'Female > Male';
    var pct = vr >= 1 ? ((vr - 1) * 100).toFixed(1) + '% more male variability'
                       : ((1 - vr) * 100).toFixed(1) + '% more female variability';
    if (Math.abs(vr - 1) < 0.005) { dir = 'Near equal'; pct = 'approximately equal variance'; }
    var out = '<strong>' + (datum.label || p.name || p.seriesName || '') + '</strong><br/>'
      + 'VR: ' + vr.toFixed(3) + '\u00d7<br/>';
    if (typeof datum.ciLow === 'number' && typeof datum.ciHigh === 'number') {
      out += '95% CI: ' + datum.ciLow.toFixed(3) + '\u00d7 to ' + datum.ciHigh.toFixed(3) + '\u00d7<br/>';
    }
    out += dir + '<br/>'
      + '<span style="color:#888">' + pct + '</span>';
    return out;
  }

  /* ── FOREST PLOT ── */
  function forest(el, data, opts) {
    opts = opts || {};
    var title = opts.title || '';
    // data: [{label, vr, ciLow?, ciHigh?}] — sorted by value
    var sorted = data.slice().sort(function (a, b) { return a.vr - b.vr; });
    var labels = sorted.map(function (d) { return d.label; });
    var pointData = sorted.map(function (d, i) {
      return {
        value: [d.vr, i],
        name: d.label,
        label: d.label,
        vr: d.vr,
        ciLow: typeof d.ciLow === 'number' ? d.ciLow : null,
        ciHigh: typeof d.ciHigh === 'number' ? d.ciHigh : null,
        itemStyle: { color: d.vr >= 1 ? null : null }
      };
    });
    var whiskerData = sorted
      .map(function (d, i) {
        if (typeof d.ciLow !== 'number' || typeof d.ciHigh !== 'number') {
          return null;
        }
        return { value: [i, d.ciLow, d.ciHigh, d.vr], label: d.label };
      })
      .filter(Boolean);
    var domainValues = [];
    sorted.forEach(function (d) {
      if (typeof d.vr === 'number') { domainValues.push(d.vr); }
      if (typeof d.ciLow === 'number') { domainValues.push(d.ciLow); }
      if (typeof d.ciHigh === 'number') { domainValues.push(d.ciHigh); }
    });
    var rawMin = domainValues.length ? Math.min.apply(null, domainValues) : 0.8;
    var rawMax = domainValues.length ? Math.max.apply(null, domainValues) : 1.2;
    var span = Math.max(0.1, rawMax - rawMin);
    var pad = Math.max(0.05, span * 0.12);
    var xMin = Math.max(0, Math.floor((Math.min(rawMin - pad, 0.98)) * 20) / 20);
    var xMax = Math.ceil((Math.max(rawMax + pad, 1.02)) * 20) / 20;

    var h = Math.max(420, sorted.length * 28 + 80);
    el.style.height = h + 'px';

    initDynamic(el, function () {
      var c = palette();
      return {
        title: title ? { text: title, left: 'center', top: 8, textStyle: { fontSize: 14, fontWeight: 600, color: c.text } } : undefined,
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: function (params) { return vrTip(params); },
          backgroundColor: c.cardBg,
          borderColor: c.border,
          textStyle: { color: c.text, fontSize: 13 }
        },
        grid: { left: 10, right: 40, top: title ? 40 : 16, bottom: 36, containLabel: true },
        xAxis: {
          type: 'value',
          min: xMin,
          max: xMax,
          name: 'Variance ratio (male / female)',
          nameLocation: 'center',
          nameGap: 24,
          nameTextStyle: { color: c.muted, fontSize: 12 },
          axisLabel: { formatter: function (v) { return v.toFixed(1) + '\u00d7'; }, color: c.muted, fontSize: 11 },
          axisLine: { lineStyle: { color: c.border } },
          splitLine: { lineStyle: { color: c.border, type: 'dashed' } }
        },
        yAxis: {
          type: 'category',
          data: labels,
          axisLabel: { color: c.text, fontSize: 11, width: 280, overflow: 'truncate' },
          axisLine: { lineStyle: { color: c.border } },
          axisTick: { show: false }
        },
        series: [
          {
            type: 'custom',
            silent: true,
            z: 1,
            data: whiskerData,
            renderItem: function (params, api) {
              var idx = api.value(0);
              var low = api.value(1);
              var high = api.value(2);
              var start = api.coord([low, idx]);
              var end = api.coord([high, idx]);
              var cap = 5;
              return {
                type: 'group',
                children: [
                  {
                    type: 'line',
                    shape: { x1: start[0], y1: start[1], x2: end[0], y2: end[1] },
                    style: { stroke: c.muted, lineWidth: 2 }
                  },
                  {
                    type: 'line',
                    shape: { x1: start[0], y1: start[1] - cap, x2: start[0], y2: start[1] + cap },
                    style: { stroke: c.muted, lineWidth: 2 }
                  },
                  {
                    type: 'line',
                    shape: { x1: end[0], y1: end[1] - cap, x2: end[0], y2: end[1] + cap },
                    style: { stroke: c.muted, lineWidth: 2 }
                  }
                ]
              };
            }
          },
          {
            type: 'scatter',
            z: 2,
            data: pointData.map(function (d) {
              return {
                value: d.value,
                name: d.name,
                label: d.label,
                ciLow: d.ciLow,
                ciHigh: d.ciHigh,
                itemStyle: { color: d.vr >= 1 ? c.pos : c.neg }
              };
            }),
            symbolSize: 10,
            emphasis: { itemStyle: { borderWidth: 2, borderColor: c.text }, scale: 1.4 },
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { color: c.muted, type: 'dashed', width: 1.5 },
              data: [{ xAxis: 1 }],
              label: { show: true, formatter: '1.0\u00d7', position: 'end', color: c.muted, fontSize: 11 }
            }
          }
        ]
      };
    });
  }

  /* ── HORIZONTAL BAR CHART ── */
  function hbar(el, data, opts) {
    opts = opts || {};
    var title = opts.title || '';
    var sorted = data.slice().sort(function (a, b) { return a.vr - b.vr; });
    var labels = sorted.map(function (d) { return d.label; });
    var values = sorted.map(function (d) { return d.vr; });

    var h = Math.max(380, sorted.length * 34 + 80);
    el.style.height = h + 'px';

    initDynamic(el, function () {
      var c = palette();
      return {
        title: title ? { text: title, left: 'center', top: 8, textStyle: { fontSize: 14, fontWeight: 600, color: c.text } } : undefined,
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: function (params) { return vrTip(params); },
          backgroundColor: c.cardBg,
          borderColor: c.border,
          textStyle: { color: c.text, fontSize: 13 }
        },
        grid: { left: 10, right: 60, top: title ? 40 : 16, bottom: 36, containLabel: true },
        xAxis: {
          type: 'value',
          name: 'Variance ratio',
          nameLocation: 'center',
          nameGap: 24,
          nameTextStyle: { color: c.muted, fontSize: 12 },
          axisLabel: { formatter: function (v) { return v.toFixed(2) + '\u00d7'; }, color: c.muted, fontSize: 11 },
          axisLine: { lineStyle: { color: c.border } },
          splitLine: { lineStyle: { color: c.border, type: 'dashed' } }
        },
        yAxis: {
          type: 'category',
          data: labels,
          axisLabel: { color: c.text, fontSize: 11, width: 260, overflow: 'truncate' },
          axisLine: { lineStyle: { color: c.border } },
          axisTick: { show: false }
        },
        series: [{
          type: 'bar',
          data: values.map(function (v, i) {
            return {
              value: v,
              name: labels[i],
              itemStyle: {
                color: v >= 1 ? c.pos : c.neg,
                borderRadius: [0, 3, 3, 0]
              },
              label: {
                show: true,
                position: 'right',
                formatter: function (p) { return p.value.toFixed(2) + '\u00d7'; },
                color: c.text,
                fontSize: 11
              }
            };
          }),
          barWidth: '60%',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: c.muted, type: 'dashed', width: 1.5 },
            data: [{ xAxis: 1 }],
            label: { show: false }
          }
        }]
      };
    });
  }

  /* ── ROBUSTNESS GROUPED BARS ── */
  function robustness(el, data) {
    var labels = data.map(function (d) { return d.label; });
    var h = Math.max(300, data.length * 50 + 100);
    el.style.height = h + 'px';

    initDynamic(el, function () {
      var c = palette();
      return {
        title: { text: 'Robustness: sensitivity to analysis variant', left: 'center', top: 8, textStyle: { fontSize: 14, fontWeight: 600, color: c.text } },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          backgroundColor: c.cardBg,
          borderColor: c.border,
          textStyle: { color: c.text, fontSize: 13 },
          formatter: function (params) {
            var out = '<strong>' + params[0].name + '</strong>';
            params.forEach(function (p) {
              out += '<br/>' + p.marker + ' ' + p.seriesName + ': ' + (p.value * 100).toFixed(1) + '%';
            });
            return out;
          }
        },
        legend: { data: ['Median |delta|', 'Sign-change rate'], bottom: 4, textStyle: { color: c.muted, fontSize: 12 } },
        grid: { left: 10, right: 30, top: 40, bottom: 48, containLabel: true },
        xAxis: {
          type: 'value',
          axisLabel: { formatter: function (v) { return (v * 100).toFixed(0) + '%'; }, color: c.muted, fontSize: 11 },
          axisLine: { lineStyle: { color: c.border } },
          splitLine: { lineStyle: { color: c.border, type: 'dashed' } }
        },
        yAxis: {
          type: 'category',
          data: labels,
          axisLabel: { color: c.text, fontSize: 11, width: 200, overflow: 'truncate' },
          axisLine: { lineStyle: { color: c.border } },
          axisTick: { show: false }
        },
        series: [
          {
            name: 'Median |delta|',
            type: 'bar',
            data: data.map(function (d) { return d.delta; }),
            itemStyle: { color: c.bar1, borderRadius: [0, 3, 3, 0] },
            barGap: '10%'
          },
          {
            name: 'Sign-change rate',
            type: 'bar',
            data: data.map(function (d) { return d.sign; }),
            itemStyle: { color: c.bar2, borderRadius: [0, 3, 3, 0] }
          }
        ]
      };
    });
  }

  /* ── AGE PROFILE SMALL MULTIPLES ── */
  function ageProfile(el, datasets) {
    // Use ECharts grid layout for small multiples
    var cols = 3;
    var rows = Math.ceil(datasets.length / cols);
    var panelH = 200;
    var h = rows * panelH + 40;
    el.style.height = h + 'px';

    initDynamic(el, function () {
      var c = palette();
      var grids = [];
      var xAxes = [];
      var yAxes = [];
      var series = [];
      var titles = [];

      datasets.forEach(function (ds, idx) {
        var col = idx % cols;
        var row = Math.floor(idx / cols);
        var left = (col / cols * 100 + 2) + '%';
        var right = ((cols - col - 1) / cols * 100 + 2) + '%';
        var top = row * panelH + 30;
        var gIdx = idx;

        grids.push({
          left: left, right: right, top: top, height: panelH - 60,
          borderColor: c.border, show: true, borderWidth: 1,
          backgroundColor: c.cardBg
        });

        titles.push({
          text: ds.name,
          left: (col / cols * 100 + (100 / cols / 2)) + '%',
          top: top - 22,
          textAlign: 'center',
          textStyle: { fontSize: 12, fontWeight: 600, color: c.text }
        });

        var points = ds.points.map(function (p) {
          return { label: p.label, vr: Math.exp(p.logvr) };
        });

        xAxes.push({
          type: 'category',
          gridIndex: gIdx,
          data: points.map(function (p) { return p.label; }),
          axisLabel: { color: c.muted, fontSize: 9, rotate: points.length > 6 ? 30 : 0 },
          axisLine: { lineStyle: { color: c.border } },
          axisTick: { show: false }
        });

        // Compute y range
        var vrs = points.map(function (p) { return p.vr; });
        var minV = Math.min.apply(null, vrs.concat([1.0]));
        var maxV = Math.max.apply(null, vrs.concat([1.0]));
        var pad = (maxV - minV) * 0.2 || 0.1;

        yAxes.push({
          type: 'value',
          gridIndex: gIdx,
          min: Math.floor((minV - pad) * 10) / 10,
          max: Math.ceil((maxV + pad) * 10) / 10,
          axisLabel: { formatter: function (v) { return v.toFixed(1) + '\u00d7'; }, color: c.muted, fontSize: 9 },
          axisLine: { lineStyle: { color: c.border } },
          splitLine: { lineStyle: { color: c.border, type: 'dashed' } }
        });

        series.push({
          type: 'line',
          xAxisIndex: gIdx,
          yAxisIndex: gIdx,
          data: points.map(function (p) {
            return {
              value: p.vr,
              itemStyle: { color: p.vr >= 1 ? c.pos : c.neg }
            };
          }),
          lineStyle: { color: c.accent, width: 2 },
          symbol: 'circle',
          symbolSize: 8,
          emphasis: { itemStyle: { borderWidth: 2, borderColor: c.text }, scale: 1.6 },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: c.muted, type: 'dashed', width: 1 },
            data: [{ yAxis: 1 }],
            label: { show: false }
          }
        });
      });

      return {
        title: titles,
        tooltip: {
          trigger: 'axis',
          formatter: function (params) {
            var p = params[0];
            return vrTip({ value: p.value, name: p.name });
          },
          backgroundColor: c.cardBg,
          borderColor: c.border,
          textStyle: { color: c.text, fontSize: 13 }
        },
        grid: grids,
        xAxis: xAxes,
        yAxis: yAxes,
        series: series
      };
    });
  }

  return { forest: forest, hbar: hbar, robustness: robustness, ageProfile: ageProfile };
})();
