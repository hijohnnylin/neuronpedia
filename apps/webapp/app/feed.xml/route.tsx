import { getPostsMetaData } from '@/app/blog/blog-util';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import RSS from 'rss';

export async function GET() {
  const blogPosts = await getPostsMetaData();

  const siteUrl = process.env.NODE_ENV === 'production' ? NEXT_PUBLIC_URL : 'http://localhost:3000';

  const feedOptions = {
    title: 'The Residual Stream',
    description: "Neuronpedia's official blog.",
    site_url: siteUrl,
    feed_url: `${siteUrl}/feed.xml`,
    image_url: `${siteUrl}/images/blog/the-residual-stream.jpg`,
    pubDate: new Date().toUTCString(),
    copyright: `All rights reserved - ${new Date().getFullYear()}`,
  };

  const feed = new RSS(feedOptions);

  blogPosts.forEach((post) => {
    feed.item({
      title: post.title,
      description: post.description,
      url: `${siteUrl}/blog/${post.slug}`,
      guid: post.slug,
      date: post.date,
    });
  });

  return new Response(feed.xml({ indent: true }), {
    headers: {
      'Content-Type': 'application/atom+xml; charset=utf-8',
    },
  });
}
