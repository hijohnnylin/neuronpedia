// TODO: there's a lot more user profile info we could show here

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/shadcn/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/shadcn/tabs';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/auth/utils';
import { getBookmarksByUserWithExplanations } from '@/lib/db/bookmark';
import { getUserByNameForPublicProfile } from '@/lib/db/user';
import { Bot } from 'lucide-react';
import { notFound } from 'next/navigation';

export default async function Page({ params }: { params: { name: string } }) {
  const authedUser = await makeAuthedUserFromSessionOrReturnNull();

  let bookmarks: any[] = [];
  const isAuthedUser = authedUser && authedUser.name === params.name;
  if (isAuthedUser && authedUser) {
    bookmarks = await getBookmarksByUserWithExplanations(authedUser);
  }

  const user = await getUserByNameForPublicProfile(params.name);
  if (!user) {
    notFound();
  }

  return (
    <div className="mt-0 flex w-full max-w-screen-xl flex-col">
      {user.bot ? (
        <>
          <div className="mt-3 rounded-full bg-violet-200 p-3">
            <Bot className="h-8 w-8 text-violet-600" />
          </div>
          <br />
          <div className="max-w-sm text-center text-sm font-medium text-slate-400">
            This user is a bot used for uploading machine-generated data, explanations, and more.
          </div>
        </>
      ) : (
        <div className="mt-5 flex w-full max-w-screen-xl flex-col items-center justify-center gap-y-5 px-3 text-xs text-slate-600">
          <Card className="min-w-80 bg-white">
            <CardHeader>
              <CardTitle>{user.name}</CardTitle>
              <CardDescription>{user.bio ? user.bio : 'No bio provided.'}</CardDescription>
            </CardHeader>
            <CardContent>
              <p>Joined {user.createdAt.toLocaleDateString()}</p>
            </CardContent>
          </Card>

          <Tabs defaultValue="bookmarks" className="mb-10 w-full max-w-screen-md bg-slate-50">
            <TabsList className="grid w-full bg-slate-50">
              {bookmarks && isAuthedUser && <TabsTrigger value="bookmarks">Private Bookmarks</TabsTrigger>}
            </TabsList>

            {bookmarks && (
              <TabsContent value="bookmarks">
                <div className="mt-2 flex max-h-[520px] max-w-screen-md flex-1 flex-col items-start justify-start overflow-y-scroll rounded-lg bg-white px-5 py-3">
                  {bookmarks && bookmarks.length > 0
                    ? bookmarks?.map((b) => (
                        <div key={b.id} className="flex flex-col rounded py-1 text-slate-500">
                          {b.neuron?.explanations && b.neuron?.explanations.length > 0 ? (
                            <div>{b.neuron?.explanations[0].description}</div>
                          ) : (
                            <div>No Explanation</div>
                          )}
                          <a
                            href={`/${b.neuron?.modelId}/${b.neuron?.layer}/${b.neuron?.index}`}
                            target="_blank"
                            rel="noreferrer"
                            className="pb-1 font-mono font-bold uppercase text-sky-700"
                          >
                            {b.neuron?.modelId}@{b.neuron?.layer}:{b.neuron?.index}
                          </a>
                        </div>
                      ))
                    : ''}
                </div>
              </TabsContent>
            )}
          </Tabs>
        </div>
      )}
    </div>
  );
}
