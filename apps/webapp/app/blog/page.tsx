import { getBlogDateString, getPostsMetaData, PostMetaData } from '@/app/blog/blog-util';
import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import { BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { Card, CardContent } from '@/components/shadcn/card';
import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import BlogSidebar from './blog-sidebar';

export async function generateMetadata(): Promise<Metadata> {
  return {
    title: `The Residual Stream - Neuronpedia's Blog`,
    description: 'The Residual Stream is Neuronpedia&apos;s blog.',
    openGraph: {
      title: `The Residual Stream - Neuronpedia's Blog`,
      description: 'The Residual Stream is Neuronpedia&apos;s blog.',
      url: `/blog`,
      images: [
        {
          url: '/images/blog/the-residual-stream.jpg',
        },
      ],
    },
  };
}

export default async function Page() {
  const posts = (await getPostsMetaData()) as PostMetaData[];

  return (
    <div className="flex min-h-full w-full flex-col items-center justify-start pb-5">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbLink href="/blog" key={1}>
            The Residual Stream - Neuronpedia&apos;s Blog
          </BreadcrumbLink>,
        ]}
      />
      <div className="flex h-full w-full max-w-screen-lg flex-col items-start justify-start px-2 pt-4 text-slate-800 sm:flex-row sm:items-center sm:justify-center sm:gap-x-7 sm:px-0">
        <BlogSidebar />

        <Card className="mt-4 h-full w-full bg-white">
          <CardContent className="w-full px-4">
            <div className="my-4 flex w-full flex-col gap-y-2 sm:gap-y-4">
              {posts &&
                posts.map((post, index) => (
                  <Link
                    href={`/blog/${post.slug}`}
                    key={index}
                    className="flex w-full flex-col items-start justify-center gap-x-4 gap-y-1 px-2 py-2 transition-all hover:bg-slate-50 sm:flex-row sm:items-center sm:px-4"
                  >
                    <Image
                      src={post.image}
                      alt={post.title}
                      width={800}
                      height={400}
                      className="w-[40%] max-w-[40%] flex-1 rounded-md"
                    />
                    <div className="flex flex-1 flex-col">
                      <p className="mb-0.5 text-sm font-bold text-slate-800 sm:mb-1 sm:text-[16px]">{post.title}</p>
                      <p className="mb-1 text-xs font-medium leading-normal text-slate-600 sm:mb-1.5 sm:text-[12px]">
                        {post.description}
                      </p>

                      <div className="flex flex-row items-center justify-between font-mono">
                        <p className="font-mono text-[11px] font-medium text-slate-600">{post.author}</p>
                        <p className="text-[11px] font-medium text-slate-600">{getBlogDateString(post.date)}</p>
                      </div>
                    </div>
                  </Link>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
