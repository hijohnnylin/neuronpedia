import { getBlogDateString, getPostBySlug, PostMetaData } from '@/app/blog/blog-util';
import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import { BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { Metadata } from 'next';
import Image from 'next/image';
import BlogSidebar from '../blog-sidebar';

type Props = {
  params: {
    slug: string;
  };
};

const getPageData = async (slug: string): Promise<{ meta: PostMetaData; content: any }> => {
  const { meta, content } = await getPostBySlug(slug);
  return { meta, content };
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { meta } = await getPageData(params.slug);
  return {
    title: { absolute: `${meta.title} | The Residual Stream` },
    description: meta.description,
    twitter: {
      card: 'summary_large_image',
      title: `${meta.title} | The Residual Stream`,
      description: meta.description,
      images: [
        {
          url: meta.imagePreview,
        },
        {
          url: '/images/blog/the-residual-stream.jpg',
        },
      ],
    },
    openGraph: {
      title: `${meta.title} | The Residual Stream`,
      description: meta.description,
      url: `/blog/${params.slug}`,
      images: [
        {
          url: meta.imagePreview,
        },
        {
          url: '/images/blog/the-residual-stream.jpg',
        },
      ],
      locale: 'en_US',
      type: 'website',
      siteName: 'Neuronpedia',
    },
  };
}

export default async function Page({ params }: Props) {
  const { meta, content } = await getPageData(params.slug);
  return (
    <div className="flex w-full flex-col items-center justify-start">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbLink href="/blog" key={1}>
            The Residual Stream - Neuronpedia&apos;s Blog
          </BreadcrumbLink>,
          <BreadcrumbLink href={`/blog/${params.slug}`} key={2}>
            {/* @ts-ignore */}
            {meta.title}
          </BreadcrumbLink>,
        ]}
      />
      <div className="flex w-full max-w-screen-lg flex-col items-center justify-start px-2 pt-4 text-slate-800 sm:flex-row sm:items-start sm:justify-center sm:gap-x-7 sm:px-0">
        <div className="sm:sticky sm:left-0 sm:top-[100px]">
          <BlogSidebar />
        </div>

        <Card className="mb-12 mt-2 w-full bg-white">
          <CardHeader className="flex flex-col px-3 pb-1 sm:px-8">
            <Image src={meta.image} alt={meta.title} width={800} height={400} className="w-full rounded-md" />
            <CardTitle className="pb-0 pt-5 text-[28px] font-bold text-slate-800">{meta.title}</CardTitle>
            <div className="mb-2 text-[15px] font-medium text-slate-800">{meta.description}</div>
            <div className="font-mono text-[12px] font-medium text-slate-600">
              By {meta.author} · {getBlogDateString(meta.date)}
            </div>
          </CardHeader>
          <CardContent className="w-full px-3 pb-6 pt-0 sm:px-8">
            <div className="prose font-sans text-slate-700">{content}</div>
            <div className="flex w-full flex-row items-center justify-between gap-x-2 pt-16">
              <a href="/blog" className="text-[14px] font-medium text-sky-700 hover:underline">
                ← Back to Blog
              </a>
              <a href="/" className="text-[14px] font-medium text-sky-700 hover:underline">
                Home
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
