import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { SteerOutputType } from '@prisma/client';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';
import { SteerResultChat } from '../steer-chat/route';

export const maxDuration = 30;

const inputSchema = object({
  steerOutputId: string().required(),
});

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();

  try {
    const body = await inputSchema.validate(bodyJson);

    const toReturnResult: SteerResultChat = {
      [SteerOutputType.STEERED]: null,
      [SteerOutputType.DEFAULT]: null,
      inputText: null,
      id: null,
      shareUrl: undefined,
      limit: '0',
      settings: undefined,
      features: undefined,
    };

    const savedSteerSteeredOutput = await prisma.steerOutput.findFirstOrThrow({
      where: {
        id: body.steerOutputId,
      },
      include: {
        connectedDefaultOutput: true,
        toNeurons: {
          include: {
            neuron: {
              select: { explanations: true, vector: true, vectorLabel: true },
            },
          },
        },
      },
    });
    toReturnResult[SteerOutputType.STEERED] = {
      raw: savedSteerSteeredOutput.outputText,
      chatTemplate: savedSteerSteeredOutput.outputTextChatTemplate
        ? JSON.parse(savedSteerSteeredOutput.outputTextChatTemplate)
        : null,
      logprobs: savedSteerSteeredOutput.logprobs ? JSON.parse(savedSteerSteeredOutput.logprobs) : null,
    };
    toReturnResult.features = savedSteerSteeredOutput.toNeurons;

    // if we can't find the connected default output, try looking it up (this only works if it's the FIRST message inthe conversation, since after the first message the two outputs diverge)
    let savedSteerDefaultOutput = savedSteerSteeredOutput.connectedDefaultOutput;
    if (!savedSteerDefaultOutput) {
      savedSteerDefaultOutput = await prisma.steerOutput.findFirst({
        where: {
          modelId: savedSteerSteeredOutput.modelId,
          type: 'DEFAULT',
          inputText: savedSteerSteeredOutput.inputText,
          temperature: savedSteerSteeredOutput.temperature,
          numTokens: savedSteerSteeredOutput.numTokens,
          freqPenalty: savedSteerSteeredOutput.freqPenalty,
          seed: savedSteerSteeredOutput.seed,
          strengthMultiplier: savedSteerSteeredOutput.strengthMultiplier,
          version: savedSteerSteeredOutput.version,
          steerSpecialTokens: savedSteerSteeredOutput.steerSpecialTokens,
          steerMethod: savedSteerSteeredOutput.steerMethod,
        },
      });
      // if we still can't find it, return an error
      if (!savedSteerDefaultOutput) {
        return NextResponse.json({ message: 'Unable to find default steer output' }, { status: 400 });
      }
    }

    toReturnResult[SteerOutputType.DEFAULT] = {
      raw: savedSteerDefaultOutput.outputText,
      chatTemplate: savedSteerDefaultOutput.outputTextChatTemplate
        ? JSON.parse(savedSteerDefaultOutput.outputTextChatTemplate)
        : null,
      logprobs: savedSteerDefaultOutput.logprobs ? JSON.parse(savedSteerDefaultOutput.logprobs) : null,
    };

    toReturnResult.inputText = savedSteerSteeredOutput.inputText;

    toReturnResult.settings = {
      temperature: savedSteerSteeredOutput.temperature,
      n_tokens: savedSteerSteeredOutput.numTokens,
      freq_penalty: savedSteerSteeredOutput.freqPenalty,
      seed: savedSteerSteeredOutput.seed,
      strength_multiplier: savedSteerSteeredOutput.strengthMultiplier,
      steer_special_tokens: savedSteerSteeredOutput.steerSpecialTokens,
      steer_method: savedSteerSteeredOutput.steerMethod as NPSteerMethod,
    };

    toReturnResult.id = savedSteerSteeredOutput.id;
    toReturnResult.shareUrl = `${env.NEXT_PUBLIC_URL}/steer/${savedSteerSteeredOutput.id}`;

    return NextResponse.json(toReturnResult);
  } catch (error) {
    console.error(error);
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
