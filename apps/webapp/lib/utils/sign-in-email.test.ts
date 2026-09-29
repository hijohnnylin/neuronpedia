import { describe, expect, it } from 'vitest';
import {
  authErrorUrl,
  INBOX_EMAILS_PER_DAY,
  INBOX_EMAILS_PER_HOUR,
  inboxKey,
  SIGN_IN_ERROR_INBOX_LIMIT,
  SIGN_IN_ERROR_SITE_CAP,
  signInEmailBlock,
  signInErrorMessage,
} from './sign-in-email';

describe('inboxKey', () => {
  it('lowercases and trims', () => {
    expect(inboxKey('  Alice@Example.COM ')).toBe('alice@example.com');
  });

  it('removes a +tag', () => {
    expect(inboxKey('alice+news@example.com')).toBe('alice@example.com');
    expect(inboxKey('alice+a+b@example.com')).toBe('alice@example.com');
  });

  it('keeps dots for domains that are not Gmail', () => {
    expect(inboxKey('a.lice@example.com')).toBe('a.lice@example.com');
    expect(inboxKey('a.lice@gmail.co.uk')).toBe('a.lice@gmail.co.uk');
  });

  it('removes dots for Gmail', () => {
    expect(inboxKey('A.Li.Ce@gmail.com')).toBe('alice@gmail.com');
    expect(inboxKey('a.lice+spam@GMAIL.com')).toBe('alice@gmail.com');
  });

  it('treats googlemail.com as gmail.com', () => {
    expect(inboxKey('a.lice@googlemail.com')).toBe('alice@gmail.com');
    expect(inboxKey('alice+x@googlemail.com')).toBe(inboxKey('ALICE@gmail.com'));
  });

  it('uses the last @ to find the domain', () => {
    expect(inboxKey('a@b@gmail.com')).toBe('a@b@gmail.com');
  });

  it('returns other input lowercased', () => {
    expect(inboxKey('NotAnEmail')).toBe('notanemail');
    expect(inboxKey('@gmail.com')).toBe('@gmail.com');
  });
});

describe('signInEmailBlock', () => {
  const base = { inboxLastHour: 1, inboxLastDay: 1, siteLastDay: 1, siteCap: 300 };

  it('allows emails under all limits', () => {
    expect(signInEmailBlock(base)).toBeNull();
    expect(
      signInEmailBlock({
        ...base,
        inboxLastHour: INBOX_EMAILS_PER_HOUR,
        inboxLastDay: INBOX_EMAILS_PER_DAY,
        siteLastDay: 300,
      }),
    ).toBeNull();
  });

  it('blocks after 3 emails to one inbox in one hour', () => {
    expect(signInEmailBlock({ ...base, inboxLastHour: INBOX_EMAILS_PER_HOUR + 1, inboxLastDay: 4 })).toBe('inbox');
  });

  it('blocks after 10 emails to one inbox in one day', () => {
    expect(signInEmailBlock({ ...base, inboxLastDay: INBOX_EMAILS_PER_DAY + 1 })).toBe('inbox');
  });

  it('blocks all emails over the site cap', () => {
    expect(signInEmailBlock({ ...base, siteLastDay: 301 })).toBe('site');
    expect(signInEmailBlock({ ...base, siteLastDay: 6, siteCap: 5, inboxLastHour: 9 })).toBe('site');
  });
});

describe('authErrorUrl', () => {
  it('makes an absolute next-auth error URL', () => {
    expect(authErrorUrl('https://www.neuronpedia.org', SIGN_IN_ERROR_SITE_CAP)).toBe(
      'https://www.neuronpedia.org/api/auth/error?error=EmailSiteCap',
    );
    expect(authErrorUrl('http://localhost:3000/api/auth', 'X')).toBe('http://localhost:3000/api/auth/error?error=X');
  });
});

describe('signInErrorMessage', () => {
  it('has a specific message for each limit', () => {
    expect(signInErrorMessage(SIGN_IN_ERROR_INBOX_LIMIT)).toMatch(/Too many/);
    expect(signInErrorMessage(SIGN_IN_ERROR_SITE_CAP)).toMatch(/paused/);
    expect(signInErrorMessage('EmailSignin')).toMatch(/could not send/);
    expect(signInErrorMessage(undefined)).toMatch(/could not send/);
  });
});
