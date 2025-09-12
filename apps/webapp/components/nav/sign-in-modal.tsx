import { useGlobalContext } from '@/components/provider/global-provider';
import Modal from '@/components/sign-in-modal/modal';
import Github from '@/components/svg/github';
import { env } from '@/lib/env';
import emailSpellChecker from '@zootools/email-spell-checker';
import { Mail } from 'lucide-react';
import { signIn } from 'next-auth/react';
import Image from 'next/image';
import { useCallback, useMemo, useState } from 'react';
import { generateFromEmail } from 'unique-username-generator';
import isEmail from 'validator/lib/isEmail';

function SignInModal() {
  const [signInClicked, setSignInClicked] = useState(false);
  const { signInModalOpen: signInOpen, setSignInModalOpen } = useGlobalContext();
  const [emailValue, setEmailValue] = useState('');

  function emailSignIn() {
    setSignInClicked(true);
    if (!isEmail(emailValue)) {
      alert('Invalid Email. Please try again.');
      setSignInClicked(false);
      return;
    }
    const suggestedEmail = emailSpellChecker.run({
      email: emailValue,
    });
    // TODO: use better dialog
    if (suggestedEmail && window.confirm(`Did you mean "${suggestedEmail.full}"?`)) {
      signIn('email', {
        email: suggestedEmail.full,
        name: generateFromEmail(suggestedEmail.full),
      });
    } else {
      signIn('email', {
        email: emailValue,
        name: generateFromEmail(emailValue),
      });
    }
  }

  return (
    <Modal showModal={signInOpen} setShowModal={setSignInModalOpen}>
      <div className="w-full overflow-hidden shadow-xl md:max-w-md md:rounded-2xl md:border md:border-slate-200">
        <div className="flex flex-col space-y-4 bg-slate-50 px-4 py-8 md:px-8">
          <div className="mb-2 flex flex-row items-center justify-center text-center text-xl font-normal text-sky-700">
            <Image
              src="/logo.png"
              alt="Neuronpedia logo - a computer chip with a rounded viewfinder border around it"
              width="32"
              height="32"
              className="mr-2"
            />
            One-Step Sign In
          </div>
          <p className="pb-1 text-center text-sm">
            Neuronpedia respects your privacy and never shares any personal information with third parties. By signing
            up, you agree to our simple{' '}
            <a href="/privacy" className="text-sky-600" target="_blank">
              Privacy Policy and Terms
            </a>
            .
          </p>
          {!env.NEXT_PUBLIC_ENABLE_SIGNIN ? (
            <div className="py-3 text-center">Not Yet Available</div>
          ) : (
            <>
              <div>
                <input
                  type="email"
                  placeholder="example@email.com"
                  value={emailValue}
                  onChange={(e) => {
                    setEmailValue(e.target.value);
                  }}
                  onKeyUp={(e) => {
                    if (e.key === 'Enter') {
                      emailSignIn();
                    }
                  }}
                  className="mb-2 w-full rounded-md border-slate-400 text-center"
                />
                <button
                  type="button"
                  disabled={signInClicked}
                  className={`${
                    signInClicked
                      ? 'cursor-not-allowed border-slate-200 bg-slate-100'
                      : 'border border-slate-200 bg-white text-black hover:bg-slate-50'
                  } flex h-10 w-full items-center justify-center space-x-1.5 rounded-md border text-sm shadow-sm transition-all duration-75 focus:outline-none`}
                  onClick={() => {
                    emailSignIn();
                  }}
                >
                  <Mail className="h-5 w-5" />
                  <p>Sign In with Email</p>
                </button>
              </div>
              <hr />
              <button
                aria-label="Sign in with Apple"
                type="button"
                disabled={signInClicked}
                className={`${
                  signInClicked ? 'cursor-not-allowed opacity-60 hover:opacity-60' : ''
                } flex h-10 w-full items-center justify-center space-x-3 rounded-md bg-black text-sm shadow-sm transition-all duration-75 hover:opacity-80 focus:outline-none`}
                onClick={() => {
                  setSignInClicked(true);
                  signIn('apple');
                }}
              >
                <div className="h-[40px] w-full bg-[url('/signin-apple.png')] bg-contain bg-center bg-no-repeat">
                  &nbsp;
                </div>
              </button>
              <button
                aria-label="Sign in with Google"
                type="button"
                disabled={signInClicked}
                className={`${
                  signInClicked ? 'cursor-not-allowed opacity-60 hover:opacity-60' : ''
                } flex h-10 w-full items-center justify-center space-x-3 rounded-md border border-slate-200 bg-white text-sm shadow-sm transition-all duration-75 focus:outline-none`}
                onClick={() => {
                  setSignInClicked(true);
                  signIn('google');
                }}
              >
                <div className="h-[38px] w-full bg-[url('/signin-google.png')] bg-contain bg-center bg-no-repeat">
                  &nbsp;
                </div>
              </button>
              <button
                aria-label="Sign in with Github"
                type="button"
                disabled={signInClicked}
                className={`${
                  signInClicked
                    ? 'cursor-not-allowed border-slate-200 bg-slate-100'
                    : 'border border-slate-200 bg-white text-black hover:bg-slate-50'
                } flex h-10 w-full items-center justify-center space-x-1.5 rounded-md border text-sm shadow-sm transition-all duration-75 focus:outline-none`}
                onClick={() => {
                  setSignInClicked(true);
                  signIn('github');
                }}
              >
                <Github className="h-5 w-5" />
                <p>Sign in with Github</p>
              </button>
            </>
          )}
        </div>
      </div>
    </Modal>
  );
}

export function useSignInModal() {
  const [showSignInModal, setShowSignInModal] = useState(false);

  const SignInModalCallback = useCallback(() => <SignInModal />, [showSignInModal, setShowSignInModal]);

  return useMemo(
    () => ({ setShowSignInModal, SignInModal: SignInModalCallback }),
    [setShowSignInModal, SignInModalCallback],
  );
}
