import { prisma } from '@/lib/db';
import { sendEmail } from '@/lib/email/email';
import { CONTACT_EMAIL_ADDRESS, NEXT_PUBLIC_URL, SIGN_IN_EMAILS_PER_DAY } from '@/lib/env';
import { inboxKey, SignInEmailBlock, signInEmailBlock } from '@/lib/utils/sign-in-email';
import { Prisma } from '@prisma/client';

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const KEEP_LOG_MS = 30 * DAY_MS;
const SITE_CAP_ALERT_LOCK = 'sign-in-email-site-cap-alert';

// Adds a log row first and then counts, so two requests at the same time both see each other.
// A blocked email does not stay in the log.
export async function reserveSignInEmail(email: string): Promise<SignInEmailBlock | null> {
  const toKey = inboxKey(email);
  const now = Date.now();
  const row = await prisma.signInEmailLog.create({ data: { to: email, toKey } });
  const [inboxLastHour, inboxLastDay, siteLastDay] = await Promise.all([
    prisma.signInEmailLog.count({ where: { toKey, sentAt: { gt: new Date(now - HOUR_MS) } } }),
    prisma.signInEmailLog.count({ where: { toKey, sentAt: { gt: new Date(now - DAY_MS) } } }),
    prisma.signInEmailLog.count({ where: { sentAt: { gt: new Date(now - DAY_MS) } } }),
  ]);
  const block = signInEmailBlock({ inboxLastHour, inboxLastDay, siteLastDay, siteCap: SIGN_IN_EMAILS_PER_DAY });

  if (block) {
    await prisma.signInEmailLog.delete({ where: { id: row.id } });
  }
  if (block === 'site') {
    console.error(
      `Sign-in email site cap reached (${SIGN_IN_EMAILS_PER_DAY} in 24 hours). Email sign-in is stopped until the count goes down.`,
    );
    await alertAdminsOfSiteCap();
  }
  await prisma.signInEmailLog.deleteMany({ where: { sentAt: { lt: new Date(now - KEEP_LOG_MS) } } });
  return block;
}

// True for only one caller in each 24 hours.
async function claimSiteCapAlert(): Promise<boolean> {
  const now = new Date();
  const expiresAt = new Date(now.getTime() + DAY_MS);
  const claimed = await prisma.processLock.updateMany({
    where: { name: SITE_CAP_ALERT_LOCK, expiresAt: { lt: now } },
    data: { startedAt: now, expiresAt },
  });
  if (claimed.count > 0) {
    return true;
  }
  try {
    await prisma.processLock.create({ data: { name: SITE_CAP_ALERT_LOCK, startedAt: now, expiresAt } });
    return true;
  } catch (error) {
    if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002') {
      return false;
    }
    throw error;
  }
}

async function alertAdminsOfSiteCap() {
  try {
    if (!(await claimSiteCapAlert())) {
      return;
    }
    const admins = await prisma.user.findMany({
      where: { admin: true, email: { not: null } },
      select: { email: true },
    });
    const recipients = admins.map((admin) => admin.email).filter((email): email is string => !!email);
    if (recipients.length === 0) {
      recipients.push(CONTACT_EMAIL_ADDRESS);
    }
    const subject = 'Neuronpedia: email sign-in is stopped (daily cap reached)';
    const html = `<body style="font-family: Helvetica, Arial, sans-serif; color: #222;">
      <p>${NEXT_PUBLIC_URL} sent ${SIGN_IN_EMAILS_PER_DAY} sign-in emails in the last 24 hours. This is the limit set by SIGN_IN_EMAILS_PER_DAY.</p>
      <p>Email sign-in is stopped until the 24-hour count goes below the limit. Other sign-in methods still work.</p>
      <p>Look at the "SignInEmailLog" table to find the addresses. You get this alert one time in 24 hours at most.</p>
    </body>`;
    await Promise.all(recipients.map((email) => sendEmail(email, undefined, subject, html)));
  } catch (error) {
    console.error('Failed to send the sign-in email site cap alert', error);
  }
}
