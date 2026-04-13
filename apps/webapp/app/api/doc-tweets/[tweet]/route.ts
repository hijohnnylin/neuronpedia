import { NextRequest, NextResponse } from 'next/server';
import { getTweet } from 'react-tweet/api';

// this is used by our docs site to get tweets
// TODO: remove this and update docs site

export const GET = async (
  request: NextRequest,
  props: {
    params: Promise<{ tweet: string }>;
  }
) => {
  const params = await props.params;
  const tweetId = params.tweet;

  if (tweetId !== '1773403396130885844' && tweetId !== '1773403397489881423' && tweetId !== '1773403398928503024') {
    return NextResponse.json({ error: 'Unsupported Tweet.' });
  }

  try {
    const tweet = await getTweet(tweetId);
    const response = NextResponse.json({ data: tweet ?? null });
    response.headers.set('Access-Control-Allow-Origin', 'https://docs.neuronpedia.org');
    response.headers.set('Access-Control-Allow-Methods', 'GET');
    response.headers.set(
      'Access-Control-Allow-Headers',
      'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
    );
    return response;
  } catch (error: any) {
    console.error(error);
    return NextResponse.json({ error: error.message ?? 'Bad request.' });
  }
};
