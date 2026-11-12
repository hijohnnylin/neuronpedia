import EmailUnsubscribeConfirm from '@/components/email-unsubscribe-confirm';
import { isValidEmailUnsubscribeCode } from '@/lib/db/user';
import { CONTACT_EMAIL_ADDRESS } from '@/lib/env';

export default async function Page(props: { searchParams?: Promise<{ [key: string]: string | undefined }> }) {
  const searchParams = await props.searchParams;
  const unsubscribeCode = searchParams?.code || '';
  const isValidCode = await isValidEmailUnsubscribeCode(unsubscribeCode);

  return (
    <EmailUnsubscribeConfirm
      code={unsubscribeCode}
      list="all"
      isValidCode={isValidCode}
      contactEmail={CONTACT_EMAIL_ADDRESS}
    />
  );
}
