// this was copied from web and I do not expect to maintain it

import type { Emphasis, InlineCode, Link, List, ListItem, Paragraph, RootContent, Strong, Text } from 'mdast';
import { fromMarkdown as mdastFromMarkdown } from 'mdast-util-from-markdown';
import { type Result, toc } from 'mdast-util-toc';

declare module 'mdast' {
  interface Node {
    key: string;
  }
}

export type ItemType = List | ListItem | Paragraph | Link | Text | InlineCode | Strong | Emphasis;

export function fromMarkdown(markdown: string): [Result, Map<string, string>] {
  const rootTree = mdastFromMarkdown(markdown);
  const result = toc(rootTree);

  const keyMap = new Map<string, string>();

  function addKey<T extends RootContent>(node: T, prefix = '') {
    node.key = prefix;
    if (node.type === 'link') {
      keyMap.set(node.url, node.key);
    }
    if ('children' in node) {
      node.children.forEach((child, index) => {
        const currentIndex = index + 1;
        addKey(child, prefix ? `${prefix}.${currentIndex}` : `${currentIndex}`);
      });
    }
  }

  if (result.map) {
    addKey(result.map);
  }
  return [result, keyMap] as const;
}
