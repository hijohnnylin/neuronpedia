import { ASSET_BASE_URL } from '@/lib/constants';
import fs from 'fs';
import { compileMDX } from 'next-mdx-remote/rsc';
import path from 'path';
import rehypeSlug from 'rehype-slug';
import remarkToc from 'remark-toc';

export type PostMetaData = {
  title: string;
  author: string;
  description: string;
  date: string;
  slug: string;
  image: string;
  imagePreview: string;
};

const rootDir = path.join(process.cwd(), 'app', 'blog', 'posts');

export function getBlogDateString(date: string) {
  return new Date(date)
    .toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric',
    })
    .replace(/(\d+)(?=(,|$))/, (match) => {
      const num = parseInt(match, 10);
      const suffix = ['th', 'st', 'nd', 'rd'][num % 100 > 10 && num % 100 < 14 ? 0 : num % 10 < 4 ? num % 10 : 0];
      return `${num}${suffix}`;
    });
}

export const getPostBySlug = async (slug: string): Promise<{ meta: PostMetaData; content: any }> => {
  const formattedSlug = slug.replace(/\.mdx$/, '');
  const filePath = path.join(rootDir, `${formattedSlug}.mdx`);
  let fileContent = fs.readFileSync(filePath, { encoding: 'utf8' });
  fileContent = fileContent.replaceAll('ASSET_BASE_URL', ASSET_BASE_URL);
  const { frontmatter, content } = await compileMDX({
    source: fileContent,
    options: {
      parseFrontmatter: true,
      mdxOptions: {
        remarkPlugins: [[remarkToc, { tight: true }]],
        rehypePlugins: [rehypeSlug],
      },
    },
  });
  return { meta: { ...frontmatter, slug: formattedSlug } as PostMetaData, content };
};

export const getPostsMetaData = async () => {
  const files = fs.readdirSync(rootDir);
  const posts = [];
  // @eslint-disable-next-line
  for (const fileName of files) {
    const { meta } = await getPostBySlug(fileName);
    posts.push(meta);
  }
  // Sort posts by date in descending order (newest first)
  posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  return posts;
};
