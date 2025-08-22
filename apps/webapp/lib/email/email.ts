import { SendEmailCommand, SESv2Client } from '@aws-sdk/client-sesv2';
import { promises as fs } from 'fs';
import _ from 'lodash';
import mjml2html from 'mjml';
import path from 'path';
import { Resend } from 'resend';
import { env } from '../env';

const { htmlToText } = require('html-to-text');

let awsSesClient: SESv2Client;
let resendClient: Resend;

if (env.AWS_ACCESS_KEY_ID && env.AWS_SECRET_ACCESS_KEY) {
  awsSesClient = new SESv2Client({
    region: 'us-east-1',
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });
} else if (env.RESEND_EMAIL_API_KEY) {
  resendClient = new Resend(env.RESEND_EMAIL_API_KEY);
}

export const EMAIL_TEMPLATES_PATH = '/lib/email';

export const EmailTemplates: {
  [key: string]: {
    subject: string;
    mjmlFile: string;
  };
} = {
  welcome: {
    subject: 'Welcome to Neuronpedia',
    mjmlFile: 'welcome.mjml',
  },
};

export const sendEmail = async (
  emailAddress: string,
  unsubscribeCode: string | undefined,
  subject: string,
  html: string,
) => {
  // try aws first
  if (awsSesClient) {
    const command = new SendEmailCommand({
      FromEmailAddress: `Neuronpedia <${env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS}>`,
      FeedbackForwardingEmailAddress: env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS,
      ConfigurationSetName: 'Neuronpedia',
      Destination: {
        ToAddresses: [emailAddress],
      },
      ReplyToAddresses: [env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS],
      Content: {
        Simple: {
          Subject: {
            Data: subject,
          },
          Body: {
            Html: {
              Data: html,
            },
            Text: {
              Data: htmlToText(html),
            },
          },
          Headers: unsubscribeCode
            ? [
                {
                  Name: 'List-Unsubscribe',
                  Value: `<${env.NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}>`,
                },
              ]
            : [],
        },
      },
    });
    const result = await awsSesClient.send(command);
    if (result.$metadata.httpStatusCode !== 200) {
      console.error('Email failed to send', result);
    }
  } else if (resendClient) {
    const result = await resendClient.emails.send({
      from: `Neuronpedia <${env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS}>`,
      to: emailAddress,
      subject,
      html,
      text: htmlToText(html),
      headers: unsubscribeCode
        ? {
            'List-Unsubscribe': `<${env.NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}>`,
          }
        : {},
    });
    if (result.error) {
      console.error('Resend email failed to send', result);
    }
  } else {
    console.error(
      'No email provider defined. Set either env.AWS_ACCESS_KEY_ID and env.AWS_SECRET_ACCESS_KEY or env.RESEND_EMAIL_API_KEY',
    );
  }
};

export const sendLoginEmail = async (emailAddress: string, signInURL: string) => {
  const buttonColor = '#0369a1';
  const subject = `Confirm Sign In to Neuronpedia`;
  const html = `<body style="background: #f9f9f9;">
      <table width="100%" border="0" cellspacing="15" cellpadding="0" style="background: #efefef; max-width: 400px; margin: auto; border-radius: 10px;">
        <tr>
          <td align="center" style="padding: 10px 0px 0px 0px; font-size: 18px; font-family: Helvetica, Arial, sans-serif; color: #444;">
            Finish signing in to Neuronpedia.
          </td>
        </tr>
        <tr>
          <td align="center" style="padding: 8px 0 16px 0;">
            <table border="0" cellspacing="0" cellpadding="0">
              <tr>
                <td align="center" style="border-radius: 5px;" bgcolor="${buttonColor}">
                  <a href="${signInURL}" target="_blank" style="font-size: 18px; font-family: Helvetica, Arial, sans-serif; text-decoration: none; border-radius: 5px; padding: 10px 20px; border: 1px solid ${buttonColor}; display: inline-block; font-weight: bold; color: white;">
                    Complete Sign In
                  </a>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td align="center" style="padding: 10px 0px 0px 0px; font-size: 15px; font-family: Helvetica, Arial, sans-serif; color: #444;">
            If that's not working, copy the link below.
          </td>
        </tr>
        <tr>
          <td align="center" style="padding: 8px 40px 16px 40px;">
            <table border="0" cellspacing="0" cellpadding="0">
              <tr>
                <td align="center" style="border-radius: 5px;">
                  <a href="${signInURL}" target="_blank" style="font-size: 14px; font-family: Helvetica, Arial, sans-serif; color: ${buttonColor}; text-decoration: none; display: inline-block; font-weight: normal;">
                    ${signInURL}
                  </a>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td align="center" style="padding: 0px 0px 10px 0px; font-size: 12px; line-height: 22px; font-family: Helvetica, Arial, sans-serif; color: #777;">
            If you didn't request this email, you can safely ignore it.
          </td>
        </tr>
      </table>
    </body>`;

  await sendEmail(emailAddress, undefined, subject, html);
};

export const sendWelcomeEmail = async (emailAddress: string, unsubscribeCode: string) => {
  const subject = 'Welcome to Neuronpedia';
  // load the welcome.mjml file
  const templatesDirectory = path.join(process.cwd(), EMAIL_TEMPLATES_PATH);
  // Check if template file exists
  try {
    await fs.access(`${templatesDirectory}/${EmailTemplates.welcome.mjmlFile}`);
  } catch (error) {
    throw new Error(
      `Welcome email template file not found at ${templatesDirectory}/${EmailTemplates.welcome.mjmlFile}`,
    );
  }
  const mjml = await fs.readFile(`${templatesDirectory}/${EmailTemplates.welcome.mjmlFile}`, 'utf8');
  // eslint-disable-next-line
  const templatedHtml = await mjml2html(mjml);
  const compiled = _.template(templatedHtml.html);
  const html = compiled({
    unsubscribe_link: `${env.NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}`,
  });
  await sendEmail(emailAddress, unsubscribeCode, subject, html);
};
