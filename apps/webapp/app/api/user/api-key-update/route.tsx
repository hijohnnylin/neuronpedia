import { removeUserSecret, updateUserSecret } from '@/lib/db/userSecret';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { UserSecretType } from '@prisma/client';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { type, value } = body;
  if (value === '') {
    await removeUserSecret(request.user.name, type);
    return NextResponse.json({ message: 'API key removed' }, { status: 200 });
  }
  if (type === UserSecretType.OPENAI) {
    try {
      const res = await fetch('https://api.openai.com/v1/models', {
        headers: { Authorization: `Bearer ${value}` },
      });
      if (!res.ok) throw new Error();
    } catch (e) {
      return NextResponse.json({ message: 'Invalid OpenAI API key' }, { status: 400 });
    }
  } else if (type === UserSecretType.OPENROUTER) {
    try {
      const res = await fetch('https://openrouter.ai/api/v1/key', {
        headers: { Authorization: `Bearer ${value}` },
      });
      if (!res.ok) throw new Error();
    } catch (e) {
      return NextResponse.json({ message: 'Invalid OpenRouter API key' }, { status: 400 });
    }
  } else if (type === UserSecretType.ANTHROPIC) {
    try {
      const res = await fetch('https://api.anthropic.com/v1/models', {
        headers: { 'x-api-key': value, 'anthropic-version': '2023-06-01' },
      });
      if (!res.ok) throw new Error();
    } catch (e) {
      return NextResponse.json({ message: 'Invalid Anthropic API key' }, { status: 400 });
    }
  } else if (type === UserSecretType.GOOGLE) {
    try {
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(value)}`,
      );
      if (!res.ok) throw new Error();
    } catch (e) {
      return NextResponse.json({ message: 'Invalid Google API key' }, { status: 400 });
    }
  }

  // valid key, update it
  await updateUserSecret(request.user.name, type, value);
  return NextResponse.json({ message: 'API key updated' }, { status: 200 });
});
