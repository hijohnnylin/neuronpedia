/* eslint-disable no-continue */

import { prisma } from '@/lib/db';
import { getActivationForFeature } from '@/lib/utils/inference';
import { ListWithPartialRelations } from '@/prisma/generated/zod';
import { AuthenticatedUser } from '../types/auth';
import { ListNeuronToAdd, MAX_LIST_FEATURES_FOR_TEST_TEXT, MAX_LIST_TEST_TEXT_LENGTH_CHARS } from '../utils/list';
import { AllowUnlistedFor, assertUserCanAccessModelAndSourceSet, userCanAccessClause } from './userCanAccess';

export const assertUserOwnsList = async (listId: string, user: AuthenticatedUser) => {
  const list = await prisma.list.findUniqueOrThrow({
    where: {
      id: listId,
    },
  });

  if (list.userId !== user.id) {
    throw new Error('User does not own list');
  } else {
    return list;
  }
};

export const newList = async (name: string, description: string, user: AuthenticatedUser) =>
  prisma.list.create({
    data: {
      name,
      description,
      userId: user.id,
    },
    include: {
      user: {
        select: {
          name: true,
        },
      },
    },
  });

export const addNeuronsToList = async (
  listId: string,
  listNeuronsToAdd: ListNeuronToAdd[],
  user: AuthenticatedUser,
) => {
  const list = await assertUserOwnsList(listId, user);

  const createdListsOnNeurons = await prisma.listsOnNeurons.createManyAndReturn({
    data: listNeuronsToAdd.map((neuronToAdd) => ({
      modelId: neuronToAdd.modelId,
      layer: neuronToAdd.layer,
      index: neuronToAdd.index,
      description: neuronToAdd.description,
      userId: user.id,
      listId,
    })),
    skipDuplicates: false,
  });

  // if there is a defaultTestText, run activations
  if (list.defaultTestText && listNeuronsToAdd.length < 20) {
    // iterate through each listNeurontoAdd
    const activationsToAdd = [];
    // eslint-disable-next-line no-restricted-syntax
    for (const neuronToAdd of createdListsOnNeurons) {
      // eslint-disable-next-line no-await-in-loop
      const neuron = await prisma.neuron.findUniqueOrThrow({
        where: {
          modelId_layer_index: {
            modelId: neuronToAdd.modelId,
            layer: neuronToAdd.layer,
            index: neuronToAdd.index,
          },
        },
        include: { source: true },
      });
      // check if the sourceSet allows activation testing
      if (neuron.source && !neuron.source.inferenceEnabled) {
        // doesn't allow it, just add without testing
        continue;
      } else {
        // does allow it, run activation
        // eslint-disable-next-line no-await-in-loop
        const activation = await getActivationForFeature(neuron, list.defaultTestText, user);
        activationsToAdd.push(activation);
      }
    }
    // create the activations
    const createdActivations = await prisma.activation.createManyAndReturn({
      data: activationsToAdd,
    });
    // connect the activations to the list
    await prisma.listsOnActivations.createMany({
      data: createdActivations.map((activation) => ({
        listId,
        activationId: activation.id,
      })),
    });
  }
  return createdListsOnNeurons;
};

export const getListWithDetails = async (listId: string, user: AuthenticatedUser | null = null) => {
  const list = (await prisma.list.findUniqueOrThrow({
    where: {
      id: listId,
    },
    include: {
      activations: {
        include: {
          activation: true,
        },
      },
      neurons: {
        where: {
          neuron: {
            model: userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
          },
        },
        orderBy: {
          addedAt: 'asc',
        },
      },
      user: {
        select: {
          name: true,
        },
      },
    },
  })) as ListWithPartialRelations;

  // get the listsOnNeurons
  // Get total count first
  const totalCount = await prisma.listsOnNeurons.count({
    where: {
      listId,
    },
  });
  let listsOnNeurons;
  if (totalCount > 300) {
    listsOnNeurons = await prisma.listsOnNeurons.findMany({
      where: {
        listId,
      },
      include: {
        neuron: {
          include: {
            activations: {
              orderBy: {
                maxValue: 'desc',
              },
              take: 3,
            },
            explanations: true,
          },
        },
      },
      take: 300,
    });
  } else {
    // Single request for smaller lists
    listsOnNeurons = await prisma.listsOnNeurons.findMany({
      where: {
        listId,
      },
      include: {
        neuron: {
          include: {
            activations: {
              orderBy: {
                maxValue: 'desc',
              },
              take: 3,
            },
            explanations: true,
          },
        },
      },
    });
  }

  list.neurons = listsOnNeurons;

  // Reorder activations to match the order of neurons
  if (list && list.activations && list.activations.length > 0 && list.neurons) {
    const neuronMap = new Map(
      list.neurons.map((neuron) => [
        `${neuron.neuron?.modelId}-${neuron.neuron?.layer}-${neuron.neuron?.index}`,
        neuron,
      ]),
    );

    list.activations.sort((a, b) => {
      const aKey = `${a.activation?.modelId}-${a.activation?.layer}-${a.activation?.index}`;
      const bKey = `${b.activation?.modelId}-${b.activation?.layer}-${b.activation?.index}`;
      const aIndex = Array.from(neuronMap.keys()).indexOf(aKey);
      const bIndex = Array.from(neuronMap.keys()).indexOf(bKey);
      return aIndex - bIndex;
    });
  }

  return list;
};

export const removeNeuronFromList = async (
  listId: string,
  modelId: string,
  layer: string,
  index: string,
  user: AuthenticatedUser,
) => {
  await assertUserOwnsList(listId, user);

  // look up all activations in the list
  const activationsInList = await prisma.listsOnActivations.findMany({
    where: {
      listId,
    },
    include: {
      activation: true,
    },
  });

  // delete the activations that match the neuron we're removing
  const activationsToDelete = activationsInList.filter(
    (listOnActivation) =>
      listOnActivation.activation.modelId === modelId &&
      listOnActivation.activation.layer === layer &&
      listOnActivation.activation.index === index,
  );

  // should only be one
  if (activationsToDelete.length > 0) {
    await prisma.activation.delete({
      where: {
        id: activationsToDelete[0].activationId,
      },
    });
  }

  // delete the neuron from the list
  const deletedListOnNeuron = await prisma.listsOnNeurons.delete({
    where: {
      modelId_layer_index_listId: {
        listId,
        modelId,
        layer,
        index,
      },
    },
  });

  return deletedListOnNeuron;
};

export const deleteList = async (listId: string, user: AuthenticatedUser) => {
  await assertUserOwnsList(listId, user);
  await prisma.listsOnNeurons.deleteMany({ where: { listId } });
  await prisma.listsOnActivations.deleteMany({ where: { listId } });
  await prisma.list.delete({ where: { id: listId } });
};

export const updateListNeuronDescription = async (
  listId: string,
  modelId: string,
  layer: string,
  index: string,
  description: string,
  user: AuthenticatedUser,
) => {
  await assertUserOwnsList(listId, user);
  return prisma.listsOnNeurons.update({
    where: {
      modelId_layer_index_listId: {
        listId,
        modelId,
        layer,
        index,
      },
    },
    data: {
      description,
    },
  });
};

export const updateListMetadata = async (
  listId: string,
  name: string,
  description: string,
  user: AuthenticatedUser,
  defaultTestText: string | null,
) => {
  await assertUserOwnsList(listId, user);

  // if there's a defaultTestText, we need to ensure we cache its activations
  if (defaultTestText) {
    const list = await prisma.list.findUniqueOrThrow({
      where: { id: listId },
      include: {
        neurons: {
          include: {
            neuron: {
              include: {
                source: true,
              },
            },
          },
          take: MAX_LIST_FEATURES_FOR_TEST_TEXT + 1,
        },
      },
    });

    // if the defaultTestText is the same, don't do anything
    if (list.defaultTestText !== defaultTestText) {
      // ensure it's not too long
      if (defaultTestText.length > MAX_LIST_TEST_TEXT_LENGTH_CHARS) {
        throw new Error(`Default test text is too long. Max is ${MAX_LIST_TEST_TEXT_LENGTH_CHARS} characters`);
      }
      // if there are default test text, ensure number of features is <= MAX_LIST_FEATURES_FOR_TEST_TEXT
      const features = list.neurons.map((neuron) => neuron.neuron).flat();
      if (features.length > MAX_LIST_FEATURES_FOR_TEST_TEXT) {
        throw new Error(
          `List has too many features to use a default test text. Max is ${MAX_LIST_FEATURES_FOR_TEST_TEXT}`,
        );
      }

      // delete existing connections to activations
      if (defaultTestText) {
        await prisma.listsOnActivations.deleteMany({
          where: { listId },
        });
        // run activations, saving each of them
        const activations = [];
        // eslint-disable-next-line no-restricted-syntax
        for (const feature of features) {
          console.log('Running activation for feature index: ', feature.index);
          if (!feature.sourceSetName) {
            console.log('No source set name for feature, skipping: ', feature);
            continue;
          }
          if (!feature.source) {
            console.log('No source for feature, skipping: ', feature);
            continue;
          }
          if (!feature.source.inferenceEnabled) {
            console.log('Source does not allow inference, skipping: ', feature);
            continue;
          }

          // eslint-disable-next-line no-await-in-loop
          await assertUserCanAccessModelAndSourceSet(feature.modelId, feature.sourceSetName, user);

          // eslint-disable-next-line no-await-in-loop
          const activation = await getActivationForFeature(feature, defaultTestText, user);

          activations.push(activation);
          // console.log("added activation: ", activation.values);
        }
        // made activations, save and connect them to the list
        // create the activations
        const createdActivations = await prisma.activation.createManyAndReturn({
          data: activations,
        });
        console.log('activations created');
        // connect the activations to the list
        await prisma.listsOnActivations.createMany({
          data: createdActivations.map((activation) => ({
            listId,
            activationId: activation.id,
          })),
        });
        console.log('activations connected to list');
      }
    }
  }

  return prisma.list.update({
    where: { id: listId },
    data: {
      name,
      description,
      defaultTestText,
    },
  });
};

export const getUserListsSimple = async (listCreatorUserId: string) =>
  prisma.list.findMany({
    where: {
      userId: listCreatorUserId,
    },
    orderBy: {
      createdAt: 'asc',
    },
  });

// MARK: Comments

export const deleteListComment = async (commentId: string, user: AuthenticatedUser) =>
  prisma.listComment.deleteMany({
    where: {
      id: commentId,
      userId: user.id,
    },
  });

export const addListComment = async (listId: string, text: string, user: AuthenticatedUser) =>
  prisma.listComment.create({
    data: {
      listId,
      text,
      userId: user.id,
    },
  });
