'use client';

// this was copied from web and I do not expect to maintain it

import _ from 'lodash';
import type { Result } from 'mdast-util-toc';
import * as React from 'react';
import { Fragment, useContext, useEffect, useRef, useState } from 'react';
import type { ItemType } from './from-markdown';

function findClosestElement(elements: Set<Element>, center: number): Element | null {
  let closestElement: Element | null = null;
  let smallestDistanceToCenter = Infinity;

  elements.forEach((element) => {
    const rect = element.getBoundingClientRect();
    const elementCenter = rect.top + rect.height / 2;
    const distanceToCenter = Math.abs(center - elementCenter);

    if (distanceToCenter < smallestDistanceToCenter) {
      smallestDistanceToCenter = distanceToCenter;
      closestElement = element;
    }
  });

  return closestElement;
}

const context = React.createContext<string>('');

export type ScrollAlign = 'start' | 'center' | 'end';

export interface TOCProviderProps {
  scrollAlign?: ScrollAlign;
  throttleTime?: number;
  keyMap: Map<string, string>;
  children: React.ReactNode;
}

export function TOCProvider(props: TOCProviderProps) {
  const { children, keyMap, scrollAlign, throttleTime = 1000 } = props;
  const [state, setState] = useState<string>('');
  const isInit = useRef(false);

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]'));
    const inViewportElements: React.MutableRefObject<Set<Element>> = {
      current: new Set(),
    };

    const setActiveKey = _.throttle(() => {
      const viewportHeight = window.innerHeight;
      const totalScrollTop = document.documentElement.scrollHeight;
      let centerForEle: number;
      if (window.scrollY < viewportHeight) {
        centerForEle = 0;
      } else if (window.scrollY + viewportHeight > totalScrollTop) {
        centerForEle = viewportHeight;
      } else {
        const viewportCenter = viewportHeight / 2;
        centerForEle = scrollAlign === 'start' ? 0 : scrollAlign === 'end' ? viewportHeight : viewportCenter;
      }
      const element = findClosestElement(inViewportElements.current, centerForEle);
      const activeId = element?.getAttribute('id');
      if (activeId) {
        const activeKey = keyMap.get(`#${activeId}`);
        if (activeKey) {
          setState(activeKey);
        }
      }
    }, throttleTime);

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            inViewportElements.current.add(entry.target);
          } else {
            inViewportElements.current.delete(entry.target);
          }
        });
        if (!isInit.current) {
          setActiveKey();
          isInit.current = true;
        }
      },
      { threshold: 0.5 },
    );

    elements.forEach((ele) => observer.observe(ele));

    window.addEventListener('scroll', setActiveKey);
    window.addEventListener('hashchange', setActiveKey);

    return () => {
      window.removeEventListener('scroll', setActiveKey);
      window.removeEventListener('hashchange', setActiveKey);

      observer.disconnect();
    };
  }, [keyMap, scrollAlign, throttleTime]);

  return <context.Provider value={state}>{children}</context.Provider>;
}

export interface LinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  activeKey: string;
  scrollAlign: ScrollAlign;
}

export function Link(props: LinkProps) {
  const { activeKey, scrollAlign, ...rest } = props;
  const currentKey = useContext(context);
  return (
    <a
      data-active={currentKey === activeKey}
      {...rest}
      onClick={(e) => {
        e.preventDefault();

        window.history.pushState(null, '', props.href);

        const id = props.href!.slice(1);
        const target = document.getElementById(id);
        target?.scrollIntoView({ behavior: 'smooth', block: scrollAlign });
      }}
    />
  );
}

export interface TOCProps {
  toc: readonly [Result, Map<string, string>];
  scrollAlign?: ScrollAlign;
  throttleTime?: number;
  renderList: RenderProps;
  renderListItem: RenderProps;
  renderLink: (children: React.ReactNode, url: string, active: boolean) => React.ReactNode;
}

type RenderProps = (children: React.ReactNode, active: boolean) => React.ReactNode;

export function TOC(props: TOCProps) {
  const {
    scrollAlign,
    throttleTime,
    toc: [result, keyMap],
    renderList,
    renderListItem,
    renderLink,
  } = props;

  function render(item: ItemType) {
    let renderFn: RenderProps | null = null;

    switch (item.type) {
      case 'list':
        renderFn = renderList;
        break;
      case 'listItem':
        renderFn = renderListItem;
        break;
      case 'link':
        renderFn = (children, active) => renderLink(children, item.url, active);
        break;
      case 'text':
      case 'inlineCode':
        return item.value;
    }
    if (renderFn) {
      return (
        <TOCImplRender key={item.key} activeKey={item.key} render={renderFn}>
          {item.children.map((child) => render(child as ItemType))}
        </TOCImplRender>
      );
    }

    return <Fragment key={item.key}>{item.children.map((child) => render(child as ItemType))}</Fragment>;
  }

  return (
    <TOCProvider throttleTime={throttleTime} scrollAlign={scrollAlign} keyMap={keyMap}>
      {result.map?.children.map((child) => render(child))}
    </TOCProvider>
  );
}

interface TOCImplRenderProps {
  activeKey: string;
  render: RenderProps;
  children: React.ReactNode;
}

function TOCImplRender(props: TOCImplRenderProps) {
  const { activeKey, render, children } = props;
  const currentKey = useContext(context);
  const active = currentKey.startsWith(activeKey);
  return render(children, active);
}
