import { env } from '@/lib/env';

export default function Privacy() {
  return (
    <div className="w-full max-w-screen-md px-3 pt-3 text-sm sm:text-base">
      <div className="my-4 text-center text-xl font-light text-slate-600 antialiased md:text-3xl lg:my-6 lg:mb-3">
        Privacy Policy & Terms of Use
      </div>
      <div className="mb-5 text-center text-slate-600">Last Updated: January 13th, 2025</div>
      <div className="mb-5 text-slate-600">
        Neuronpedia is built and operated by Decode Research and is fiscally sponsored by Ashgro Inc. Its primary
        contributor is{' '}
        <a href="https://johnnylin.co" target="_blank" rel="noreferrer" className="text-sky-500">
          Johnny Lin
        </a>
        , who cares about privacy and understands it well. He previously founded a privacy startup and has written about
        it in places like{' '}
        <a
          href="https://www.fastcompany.com/90591586/apple-privacy-nutrition-labels-flaws"
          target="_blank"
          rel="noreferrer"
          className="text-sky-500"
        >
          FastCompany
        </a>{' '}
        and{' '}
        <a
          href="https://www.washingtonpost.com/technology/2021/09/23/iphone-tracking/"
          target="_blank"
          rel="noreferrer"
          className="text-sky-500"
        >
          WaPo
        </a>
        .
      </div>
      <div id="privacy" className="mt-8 text-center text-lg font-light text-slate-600 antialiased md:text-2xl">
        Privacy
      </div>

      <div className="pt-3 text-slate-600">
        Neuronpedia is a new project that aims to accelerate AI interpretability research. It currently has no method or
        plans for monetization. There are no ads, third party trackers, or tracking across different websites.
        <br />
        <br />
        That said, the point of Neuronpedia is for users to contribute useful data to help explain, understand, and
        build safer AI. In order to do that, Neuronpedia does need to collect said data, and do things like user
        registration so you can log in. Neuronpedia uses{' '}
        <a href="https://github.com/vercel/analytics" className="text-sky-600">
          anonymized analytics
        </a>{' '}
        that&apos;s built into the cloud provider, which gives it broad, non-personally identifiable information about
        things like how many people visit the website. Neuronpedia also uses Sentry for crash reporting. Sentry does not
        collect any personally identifiable information.
        <br />
        <br />
        If you log in via Github, Apple, or Google, Neuronpedia saves the email address that you have provided to those
        accounts in order to keep track of your account and send you transactional emails, like emails to reset your
        password. Neuronpedia uses cookies to ensure you stay logged in. Neuronpedia does not request any permissions to
        your Github repositories. If you are feeling paranoid, feel free to create a throwaway Github account to log in.
        However, please note that Neuronpedia cannot merge accounts for you later if you do wish to credit your actual
        Github account with your Neuronpedia contributions.
        <br />
        <br />
        When you contribute to Neuronpedia, such as vote on explanations, or add new explanations for a neuron, your
        public username may be publicly credited with that contribution. You may change your username at any time. If
        you wish to delete your data, you can simply{' '}
        <a
          href={`mailto:${env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS}?subject=Deletion%20Request&body=Hey%2C%0D%0A%0D%0APlease%20delete%20my%20account.%0D%0AThe%20email%20associated%20with%20my%20account%20is%3A%20%5Bthis%20should%20match%20the%20address%20you're%20emailing%20from%5D%0D%0AMy%20username%20is%3A%20%5Busername%5D%0D%0A%0D%0AThank%20you!%0D%0A`}
          className="text-sky-500"
        >
          email us your deletion request
        </a>{' '}
        from the email address that you registered with.
        <br />
        <br />
        Neuronpedia caches certain data and results in order to improve response time and avoid strain on our GPUs. For
        example, when you do a Search By Inference, a steer, or a Top-K Search, we save the new activations data to our
        database, so that when you reload this page later, or when you share it, we can just serve it instantly without
        accessing the GPU again. The alternative would require an additional ~10 second loading time, and much longer if
        our GPUs are occupied. Your searches and steers are not browsable by anyone else, unless you specifically share
        the link with them. One-time activation tests on a feature are not cached, since they are GPU-light.
        <br />
        <br />
        In order to provide AI researchers with useful data, Neuronpedia will allow exports of the data you contribute.
        For example, if you add an explanation for a neuron, your explanation may later be exported for analysis by
        researchers. Remember, that is the entire point of the website - to advance AI interpretability. Neuronpedia
        will not include personally identifiable information in these data exports, like your email address or IP
        address.
      </div>
      <div id="terms" className="mt-8 text-center text-lg font-light text-slate-600 antialiased md:text-2xl">
        Terms of Use
      </div>
      <div className="pb-8 pt-3 text-left text-slate-600">
        <strong>Don&apos;t do illegal stuff.</strong> We&apos;re not sure how you could use Neuronpedia to do anything
        illegal, but if you figure out a way, don&apos;t do it. Illegal means whatever the laws prohibit in your
        jurisdiction and Neuronpedia&apos;s (California, USA). This includes things like spamming Neuronpedia or trying
        to abuse the voting system.
        <br />
        <br />
        <strong>Don&apos;t be mean.</strong> Other people are also on Neuronpedia, and they presumably want to have an
        enjoyable experience while helping advance AI interpretability. If you are abusive, mean, or harmful to other
        users, you will be banned. Neuronpedia reserves the right to define harmful behavior.
        <br />
        <br />
        <strong>Warranty.</strong> This website comes with zero warranty and by using it, you agree to not hold
        Neuronpedia or its creators liable for anything that happens on it. Neuronpedia does not guarantee that anything
        will work perfectly or even correctly. If you use Neuronpedia, you are responsible for your actions on
        Neuronpedia. Neuronpedia will not protect you if you do silly things like try to harass another user.
      </div>
      <div id="questions" className="mt-0 text-center text-lg font-light text-slate-600 antialiased md:text-2xl">
        Questions?
      </div>
      <div className="mb-16 pb-16 pt-3 text-left text-slate-600">
        Is this missing something that you&apos;d like to know?{' '}
        <a
          href="mailto:johnny@neuronpedia.org?subject=Privacy%20%26%20Terms%20Question%2FComment&body=Hey%20Johnny%2C%0D%0A%0D%0AHere's%20my%20question%20or%20comment%20about%20the%20privacy%20%2F%20terms%3A%0D%0A%0D%0A%0D%0A%0D%0AThank%20you!%0D%0A"
          className="text-sky-500"
        >
          Send us your question or comment
        </a>
        .
      </div>
    </div>
  );
}
