import { SendEmailCommand, SESv2Client } from '@aws-sdk/client-sesv2';
import { promises as fs } from 'fs';
import _ from 'lodash';
import mjml2html from 'mjml';
import * as nodemailer from 'nodemailer';
import path from 'path';
import { Resend } from 'resend';
import {
  AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY,
  CONTACT_EMAIL_ADDRESS,
  EMAIL_SENDING_PROVIDER,
  NEXT_PUBLIC_URL,
  RESEND_EMAIL_API_KEY,
  SMTP_SERVER_HOST,
  SMTP_SERVER_PORT,
} from '../env';

const { htmlToText } = require('html-to-text');

let awsSesClient: SESv2Client;
let resendClient: Resend;
let smtpTransporter: nodemailer.Transporter;

switch (EMAIL_SENDING_PROVIDER) {
  case 'aws':
    awsSesClient = new SESv2Client({
      region: 'us-east-1',
      credentials: {
        accessKeyId: AWS_ACCESS_KEY_ID,
        secretAccessKey: AWS_SECRET_ACCESS_KEY,
      },
    });
    break;
  case 'resend':
    resendClient = new Resend(RESEND_EMAIL_API_KEY);
    break;
  case 'smtp':
    smtpTransporter = nodemailer.createTransport({
      host: SMTP_SERVER_HOST,
      port: Number(SMTP_SERVER_PORT),
      secure: false,
      requireTLS: true,
    });
    break;
  default:
    console.error('No email provider defined. Set EMAIL_SENDING_PROVIDER');
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
  switch (EMAIL_SENDING_PROVIDER) {
    case 'aws': {
      const command = new SendEmailCommand({
        FromEmailAddress: `Neuronpedia <${CONTACT_EMAIL_ADDRESS}>`,
        FeedbackForwardingEmailAddress: CONTACT_EMAIL_ADDRESS,
        ConfigurationSetName: 'Neuronpedia',
        Destination: {
          ToAddresses: [emailAddress],
        },
        ReplyToAddresses: [CONTACT_EMAIL_ADDRESS],
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
                    Value: `<${NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}>`,
                  },
                ]
              : [],
          },
        },
      });
      const awsResult = await awsSesClient.send(command);
      if (awsResult.$metadata.httpStatusCode !== 200) {
        console.error('Email failed to send', awsResult);
      }
      break;
    }

    case 'resend': {
      const resendDotComResult = await resendClient.emails.send({
        from: `Neuronpedia <${CONTACT_EMAIL_ADDRESS}>`,
        to: emailAddress,
        subject,
        html,
        text: htmlToText(html),
        headers: unsubscribeCode
          ? {
              'List-Unsubscribe': `<${NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}>`,
            }
          : {},
      });
      if (resendDotComResult.error) {
        console.error('Resend email failed to send', resendDotComResult);
      }
      break;
    }

    case 'smtp': {
      try {
        const mailOptions: nodemailer.SendMailOptions = {
          from: `Neuronpedia <${CONTACT_EMAIL_ADDRESS}>`,
          to: emailAddress,
          subject,
          html,
          text: htmlToText(html),
          headers: unsubscribeCode
            ? {
                'List-Unsubscribe': `<${NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}>`,
              }
            : undefined,
        };
        await smtpTransporter.sendMail(mailOptions);
      } catch (error) {
        console.error('SMTP email failed to send', error);
      }
      break;
    }

    default:
      console.error('No email provider defined. Set EMAIL_SENDING_PROVIDER');
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
    unsubscribe_link: `${NEXT_PUBLIC_URL}/unsubscribe-all?code=${unsubscribeCode}`,
  });
  await sendEmail(emailAddress, unsubscribeCode, subject, html);
};
