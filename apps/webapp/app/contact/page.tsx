import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import { BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/shadcn/card';

export default async function Page() {
  return (
    <div className="flex min-h-full w-full flex-col items-center justify-start pb-5">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbLink href="/contact" key={1}>
            Contact
          </BreadcrumbLink>,
        ]}
      />
      <div className="flex h-full w-full max-w-screen-md flex-col items-start justify-start gap-y-2 px-2 pt-4 text-slate-700">
        <Card className="mt-8 w-full bg-white">
          <CardHeader>
            <CardTitle className="text-center text-xl">Support & Contact</CardTitle>
            <CardDescription className="mt-3 pt-2 text-sm">
              Neuronpedia is constantly evolving for the latest in interpretability science, so your feedback, feature
              requests, and bug reports are critical to its success. Here&apos;s how to reach us:
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-y-2">
            <p>
              ·{' '}
              <a
                href="https://github.com/hijohnnylin/neuronpedia/issues"
                className="font-bold text-sky-700 hover:underline"
              >
                GitHub Issues
              </a>{' '}
              - Good for longer-form feature requests and specific bugs.
            </p>
            <p>
              ·{' '}
              <a
                href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3m2fulfeu-0LnVnF8yCrKJYQvWLuCQaQ"
                className="font-bold text-sky-700 hover:underline"
              >
                Slack #neuronpedia or #general
              </a>{' '}
              - Good for quick questions and general tips.
            </p>
            <p>
              ·{' '}
              <a href="mailto:johnny@neuronpedia.org" className="font-bold text-sky-700 hover:underline">
                Email
              </a>{' '}
              - Good for high priority issues, privacy-sensitive communication, and collaborations.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
