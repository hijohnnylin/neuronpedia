import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const originalPayload = await request.json();
    const apiKey = request.headers.get('X-Forwarded-Client-API-Key');

    const targetUrl = 'https://www.neuronpedia.org/api/steer-chat';
    const externalHeaders: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (apiKey) {
      externalHeaders['X-API-Key'] = apiKey;
    }

    const externalResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: externalHeaders,
      body: JSON.stringify(originalPayload),
    });

    // Try to parse JSON, but if it fails, it might be a non-JSON error response
    let data;
    try {
      data = await externalResponse.json();
    } catch (e) {
      // If JSON parsing fails, try to get text, it might be an HTML error page or plain text
      const textError = await externalResponse.text();
      // If the original request was not ok, and we couldn't parse JSON, return the text error
      if (!externalResponse.ok) {
        return NextResponse.json({ error: 'Failed to proxy request', details: textError }, { status: externalResponse.status || 500 });
      }
      // If it was ok, but not JSON (unlikely for an API), this is an unexpected situation
      return NextResponse.json({ error: 'Unexpected response format from external API', details: textError }, { status: 500 });
    }

    if (!externalResponse.ok) {
      // Forward the error response from the external API (already parsed as JSON)
      return NextResponse.json(data, { status: externalResponse.status });
    }

    return NextResponse.json(data, { status: externalResponse.status });

  } catch (error: any) {
    console.error('Proxy error:', error);
    // Check if the error is a TypeError from request.json() failing (e.g. empty body for GET)
    // Though for POST, a body is expected. This is more a general guard.
    if (error instanceof TypeError && error.message.includes('body stream already read') || error.message.includes('JSON Parse error')) {
        return NextResponse.json({ error: 'Invalid request payload' }, { status: 400 });
    }
    return NextResponse.json({ error: 'Internal Server Error during proxying' }, { status: 500 });
  }
}