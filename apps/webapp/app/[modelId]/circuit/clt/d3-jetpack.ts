// @ts-nocheck
/* eslint-disable */

// typescript fork of anthropic's fork of d3-jetpack:
// https://transformer-circuits.pub/2025/attribution-graphs/static_js/lib/jetpack_2024-07-20.js
// claude 3.7 wrote this, so uhh ¯\_(ツ)_/¯
//
// Forked from https://github.com/gka/d3-jetpack — BSD-3

import * as d3 from 'd3';

// Extend d3 types
declare module 'd3' {
  export interface Selection<GElement extends d3.BaseType, Datum, PElement extends d3.BaseType, PDatum> {
    selectAppend: (name: string) => Selection<d3.BaseType, Datum, PElement, PDatum>;
    appendMany: <NewDatum>(name: string, data: NewDatum[]) => Selection<d3.BaseType, NewDatum, PElement, PDatum>;
    at: (name: string | object, value?: any) => Selection<GElement, Datum, PElement, PDatum>;
    st: (name: string | object, value?: any) => Selection<GElement, Datum, PElement, PDatum>;
    translate: (
      xy: [number, number] | ((d: any, i: number) => [number, number]),
      dim?: number,
    ) => Selection<GElement, Datum, PElement, PDatum>;
    parent: () => Selection<d3.BaseType, Datum, null, undefined>;
  }

  export function nestBy<T>(array: T[], key: (d: T) => string): Array<T[] & { key: string }>;
  export function clamp(min: number, d: number, max: number): number;
  export function conventions(c?: any): any;
  export function attachTooltip(sel: any, tooltipSel?: any, fieldFns?: any): void;
}

// Helper function to parse attributes from tag names like "div.class#id"
function parseAttributes(name: string | any): {
  tag: string;
  attr: Record<string, any>;
} {
  if (typeof name === 'string') {
    const attr: Record<string, any> = {};
    const parts = name.split(/([\.#])/g);
    let p: string;

    name = parts.shift() || '';
    while ((p = parts.shift() as string)) {
      if (p == '.') attr.class = attr.class ? `${attr.class} ${parts.shift()}` : parts.shift();
      else if (p == '#') attr.id = parts.shift();
    }
    return { tag: name, attr };
  }
  return name;
}

// Add selectAppend method to d3.selection
d3.selection.prototype.selectAppend = function (name) {
  const select = d3.selector(name);
  const n = parseAttributes(name);
  const creator = d3.creator(n.tag);

  const s = this.select(function () {
    return select.apply(this, arguments) || this.appendChild(creator.apply(this, arguments));
  });

  for (const key in n.attr) {
    s.attr(key, n.attr[key]);
  }
  return s;
};

// Add appendMany method to d3.selection
d3.selection.prototype.appendMany = function (name, data) {
  return this.selectAll(name).data(data).join(name);
};

// Override append method
d3.selection.prototype.append = function (name) {
  let create;
  let n;

  if (typeof name === 'function') {
    create = name;
  } else {
    n = parseAttributes(name);
    create = d3.creator(n.tag);
  }

  const sel = this.select(function () {
    return this.appendChild(create.apply(this, arguments));
  });

  if (n) {
    for (const key in n.attr) {
      sel.attr(key, n.attr[key]);
    }
  }
  return sel;
};

// Define regex for camelCase attributes
const camelCaseAttrs =
  /^(alignmentBaseline|allowReorder|attributeName|attributeType|autoReverse|baseFrequency|baselineShift|baseProfile|calcMode|clipPathUnits|clipRule|colorInterpolation|colorInterpolationFilters|colorProfile|contentScriptType|contentStyleType|diffuseConstant|dominantBaseline|edgeMode|enableBackground|externalResourcesRequired|fillOpacity|fillRule|filterRes|filterUnits|floodColor|floodOpacity|fontFamily|fontSize|fontSizeAdjust|fontStretch|fontStyle|fontVariant|fontWeight|glyphOrientationHorizontal|glyphOrientationVertical|glyphRef|gradientTransform|gradientUnits|imageRendering|kernelMatrix|kernelUnitLength|kerning|keyPoints|keySplines|keyTimes|lengthAdjust|letterSpacing|lightingColor|limitingConeAngle|markerEnd|markerMid|markerStart|maskContentUnits|maskUnits|midMarker|numOctaves|overlinePosition|overlineThickness|paintOrder|pathLength|patternContentUnits|patternTransform|patternUnits|pointerEvents|pointsAtX|pointsAtY|pointsAtZ|preserveAlpha|preserveAspectRatio|primitiveUnits|referrerPolicy|repeatCount|repeatDur|requiredExtensions|requiredFeatures|shapeRendering|specularConstant|specularExponent|spreadMethod|startOffset|stdDeviation|stopColor|stopOpacity|stitchTiles|strikethroughPosition|strikethroughThickness|strokeDasharray|strokeDashoffset|strokeLinecap|strokeLinejoin|strokeMiterlimit|strokeOpacity|strokeWidth|surfaceScale|systemLanguage|tableValues|targetX|targetY|textAnchor|textDecoration|textLength|textRendering|underlinePosition|underlineThickness|vectorEffect|viewTarget|wordSpacing|writingMode|xChannelSelector|yChannelSelector|zoomAndPan)$/;

// Add at method for attributes
d3.selection.prototype.at = function (name, value) {
  if (typeof name === 'object') {
    for (const key in name) {
      this.attr(camelCaseAttrs.test(key) ? key.replace(/([a-z\d])([A-Z])/g, '$1-$2').toLowerCase() : key, name[key]);
    }
    return this;
  }
  return arguments.length == 1 ? this.attr(name) : this.attr(name, value);
};

// Add st method for styling
d3.selection.prototype.st = function (name, value) {
  function addPx(d: any) {
    return d.match ? d : `${d}px`;
  }

  function wrapPx(fn: Function) {
    return function () {
      const val = fn.apply(this, arguments);
      return addPx(val);
    };
  }

  function addStyle(sel: any, style: string, value: any) {
    style = style.replace(/([a-z\d])([A-Z])/g, '$1-$2').toLowerCase();

    const pxStyles =
      'top left bottom right padding-top padding-left padding-bottom padding-right border-top border-left-width border-bottom-width border-right-width margin-top margin-left margin-bottom margin-right font-size width stroke-width line-height margin padding border border-radius max-width min-width max-height min-height gap';

    if (pxStyles.indexOf(style) >= 0) {
      sel.style(style, typeof value === 'function' ? wrapPx(value) : addPx(value));
    } else {
      sel.style(style, value);
    }

    return sel;
  }

  if (typeof name === 'object') {
    for (const key in name) {
      addStyle(this, key, name[key]);
    }
    return this;
  }
  return arguments.length == 1 ? this.style(name) : addStyle(this, name as string, value);
};

// Add translate method
d3.selection.prototype.translate = function (xy, dim) {
  const node = this.node();
  return !node
    ? this
    : node.getBBox
      ? this.attr('transform', function (d, i) {
          const p = typeof xy === 'function' ? xy.call(this, d, i) : xy;
          if (dim === 0) return `translate(${p},0)`;
          if (dim === 1) return `translate(0,${p})`;
          return `translate(${p[0]},${p[1]})`;
        })
      : this.style('transform', function (d, i) {
          const p = typeof xy === 'function' ? xy.call(this, d, i) : xy;
          if (dim === 0) return `translate(${p}px,0px)`;
          if (dim === 1) return `translate(0px,${p}px)`;
          return `translate(${p[0]}px,${p[1]}px)`;
        });
};

// Add parent method
d3.selection.prototype.parent = function () {
  const parents: any[] = [];
  return this.filter(function () {
    if (parents.indexOf(this.parentNode) > -1) return false;
    parents.push(this.parentNode);
    return true;
  }).select(function () {
    return this.parentNode;
  });
};

// Add nestBy function
(d3 as any).nestBy = function <T>(array: T[], key: (d: T) => string): Array<T[] & { key: string }> {
  return d3.groups(array, key).map(([key, values]) => {
    (values as any).key = key;
    return values as any;
  });
};

// Add clamp function
(d3 as any).clamp = function (min: number, d: number, max: number): number {
  return Math.max(min, Math.min(max, d));
};

// Add conventions function
(d3 as any).conventions = function (c?: any) {
  c = c || {};

  // ensure we're running in the browser
  if (typeof window !== 'undefined') {
    c.margin = c.margin || {};
    ['top', 'right', 'bottom', 'left'].forEach((d) => {
      if (!c.margin[d] && c.margin[d] !== 0) c.margin[d] = 20;
    });

    if (c.parentSel) c.sel = c.parentSel; // backwards compat
    const node = c.sel && c.sel.node();

    c.totalWidth = c.totalWidth || (node && node.offsetWidth) || 960;
    c.totalHeight = c.totalHeight || (node && node.offsetHeight) || 500;

    c.width = c.width || c.totalWidth - c.margin.left - c.margin.right;
    c.height = c.height || c.totalHeight - c.margin.top - c.margin.bottom;

    c.totalWidth = c.width + c.margin.left + c.margin.right;
    c.totalHeight = c.height + c.margin.top + c.margin.bottom;

    c.sel = c.sel || d3.select('body');
    c.sel.st({
      position: 'relative',
      height: c.totalHeight,
      width: c.totalWidth,
    });

    c.x = c.x || d3.scaleLinear().range([0, c.width]);
    c.y = c.y || d3.scaleLinear().range([c.height, 0]);

    c.xAxis = c.xAxis || d3.axisBottom().scale(c.x);
    c.yAxis = c.yAxis || d3.axisLeft().scale(c.y);

    c.layers = (c.layers || 's').split('').map((type) => {
      let layer;
      if (type == 's') {
        layer = c.sel
          .append('svg')
          .st({ position: c.layers ? 'absolute' : '' })
          .attr('width', c.totalWidth)
          .attr('height', c.totalHeight)
          .append('g')
          .attr('transform', `translate(${c.margin.left},${c.margin.top})`);

        if (!c.svg) c.svg = layer; // defaults to lowest svg layer
      } else if (type == 'c') {
        const s = window.devicePixelRatio || 1;

        layer = c.sel
          .append('canvas')
          .at({ width: c.totalWidth * s, height: c.totalHeight * s })
          .st({ width: c.totalWidth, height: c.totalHeight })
          .st({ position: 'absolute' })
          .node()
          .getContext('2d');
        layer.scale(s, s);
        layer.translate(c.margin.left, c.margin.top);
      } else if (type == 'd') {
        layer = c.sel.append('div').st({
          position: 'absolute',
          left: c.margin.left,
          top: c.margin.top,
          width: c.width,
          height: c.height,
        });
      }

      return layer;
    });

    c.drawAxis = (svg?: any) => {
      if (!svg) svg = c.svg;
      const xAxisSel = svg
        .append('g')
        .attr('class', 'x axis')
        .attr('transform', `translate(0,${c.height})`)
        .call(c.xAxis);

      const yAxisSel = svg.append('g').attr('class', 'y axis').call(c.yAxis);

      return { xAxisSel, yAxisSel };
    };
  }
  return c;
};

// Add attachTooltip function
(d3 as any).attachTooltip = function (sel: any, tooltipSel?: any, fieldFns?: any) {
  if (!sel.size()) return;

  tooltipSel = tooltipSel || d3.select('.tooltip');

  sel
    .on('mouseover.attachTooltip', ttDisplay)
    .on('mousemove.attachTooltip', ttMove)
    .on('mouseout.attachTooltip', ttHide)
    .on('click.attachTooltip', (e: any, d: any) => console.log(d));

  const d = sel.datum();
  fieldFns =
    fieldFns ||
    Object.keys(d)
      .filter((str) => typeof d[str] !== 'object' && d[str] != 'array')
      .map(
        (str) =>
          function (d: any) {
            return `${str}: <b>${d[str]}</b>`;
          },
      );

  function ttDisplay(d: any) {
    tooltipSel
      .classed('tooltip-hidden', false)
      .html('')
      .appendMany('div', fieldFns)
      .html((fn: Function) => fn(d));

    d3.select(this).classed('tooltipped', true);
  }

  function ttMove(e: any) {
    if (!tooltipSel.size()) return;

    const x = e.clientX;
    const y = e.clientY;
    const bb = tooltipSel.node().getBoundingClientRect();
    const left = (d3 as any).clamp(20, x - bb.width / 2, window.innerWidth - bb.width - 20);
    const top = window.innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

    tooltipSel.style('left', `${left}px`).style('top', `${top}px`);
  }

  function ttHide() {
    tooltipSel.classed('tooltip-hidden', true);
    d3.selectAll('.tooltipped').classed('tooltipped', false);
  }
};

export default d3;
