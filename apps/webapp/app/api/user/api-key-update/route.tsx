import { OPENROUTER_BASE_URL } from '@/lib/constants';
import { removeUserSecret, updateUserSecret } from '@/lib/db/userSecret';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { UserSecretType } from '@prisma/client';
import { NextResponse } from 'next/server';
import OpenAI from 'openai';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { type, value } = body;
  if (value === '') {
    await removeUserSecret(request.user.name, type);
    return NextResponse.json({ message: 'API key removed' }, { status: 200 });
  }
  if (type === UserSecretType.OPENAI || type === UserSecretType.OPENROUTER) {
    // test the openai api key
    try {
      const openai = new OpenAI({
        baseURL: type === UserSecretType.OPENAI ? undefined : OPENROUTER_BASE_URL,
        apiKey: value,
      });
      await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        temperature: 0,
      });
    } catch (e) {
      return NextResponse.json({ message: 'Invalid OpenAI or OpenRouter API key' }, { status: 400 });
    }
  } else if (type === UserSecretType.ANTHROPIC) {
    try {
      const anthropic = new Anthropic({ apiKey: value });
      await anthropic.messages.create({
        model: 'claude-3-5-sonnet-20240620',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        temperature: 0,
      });
    } catch (e) {
      return NextResponse.json({ message: 'Invalid Anthropic API key' }, { status: 400 });
    }
  } else if (type === UserSecretType.GOOGLE) {
    try {
      const gemini = new GoogleGenerativeAI(value);
      const model = gemini.getGenerativeModel({
        model: 'gemini-1.5-pro',
      });
      const chat = model.startChat({
        history: [],
      });
      const result = await chat.sendMessage('hi');
      result.response.text();
    } catch (e) {
      return NextResponse.json({ message: 'Invalid Google API key' }, { status: 400 });
    }
  }

  // valid key, update it
  await updateUserSecret(request.user.name, type, value);
  return NextResponse.json({ message: 'API key updated' }, { status: 200 });
});
