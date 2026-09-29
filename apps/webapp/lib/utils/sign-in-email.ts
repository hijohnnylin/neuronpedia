// Limits and error codes for sign-in (magic link) emails. No server imports: client code uses this file too.

export const INBOX_EMAILS_PER_HOUR = 3;
export const INBOX_EMAILS_PER_DAY = 10;

// Values of `error` in the URL that next-auth gives back to `signIn()`.
export const SIGN_IN_ERROR_INBOX_LIMIT = 'EmailInboxLimit';
export const SIGN_IN_ERROR_SITE_CAP = 'EmailSiteCap';
export const SIGN_IN_ERROR_TURNSTILE = 'TurnstileFailed';

const GMAIL_DOMAINS = new Set(['gmail.com', 'googlemail.com']);

// One key for all addresses that go to the same inbox: lowercase, no "+tag", and no dots for Gmail.
export function inboxKey(email: string): string {
  const address = email.trim().toLowerCase();
  const at = address.lastIndexOf('@');
  if (at <= 0) {
    return address;
  }
  let local = address.slice(0, at);
  let domain = address.slice(at + 1);
  const plus = local.indexOf('+');
  if (plus >= 0) {
    local = local.slice(0, plus);
  }
  if (GMAIL_DOMAINS.has(domain)) {
    domain = 'gmail.com';
    local = local.replace(/\./g, '');
  }
  return `${local}@${domain}`;
}

export type SignInEmailBlock = 'inbox' | 'site';

// Counts include the email that we want to send now.
export function signInEmailBlock(counts: {
  inboxLastHour: number;
  inboxLastDay: number;
  siteLastDay: number;
  siteCap: number;
}): SignInEmailBlock | null {
  if (counts.siteLastDay > counts.siteCap) {
    return 'site';
  }
  if (counts.inboxLastHour > INBOX_EMAILS_PER_HOUR || counts.inboxLastDay > INBOX_EMAILS_PER_DAY) {
    return 'inbox';
  }
  return null;
}

// Must be absolute: the next-auth client reads `error` with `new URL()`.
export function authErrorUrl(baseUrl: string, error: string): string {
  const url = new URL('/api/auth/error', baseUrl);
  url.searchParams.set('error', error);
  return url.toString();
}

export function signInErrorMessage(error: string | null | undefined): string {
  switch (error) {
    case SIGN_IN_ERROR_INBOX_LIMIT:
      return 'Too many sign-in emails were sent to this address. Wait one hour and try again, or use a different sign-in method.';
    case SIGN_IN_ERROR_SITE_CAP:
      return 'Email sign-in is paused for now. Try again later, or use a different sign-in method.';
    case SIGN_IN_ERROR_TURNSTILE:
      return 'The security check failed. Try again.';
    default:
      return 'We could not send the sign-in email. Try again.';
  }
}
