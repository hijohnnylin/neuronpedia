import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const apiKey = request.headers.get('X-API-Key');

    const targetUrl = 'https://www.neuronpedia.org/api/steer-chat';

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }

    const externalResponse = await fetch(targetUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    const responseData = await externalResponse.json();

    if (!externalResponse.ok) {
      return NextResponse.json(responseData, { status: externalResponse.status });
    }

    return NextResponse.json(responseData, { status: externalResponse.status });
  } catch (error) {
    console.error('Proxy error:', error);
    if (error instanceof Error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ error: 'An unknown error occurred' }, { status: 500 });
  }
}