import { useGlobalContext } from '@/components/provider/global-provider';
import * as Dialog from '@radix-ui/react-dialog';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Select from '@radix-ui/react-select';
import { Field, Form, Formik } from 'formik';
import { Check } from 'lucide-react';
import { useSession } from 'next-auth/react';
import {
  ListsOnNeuronsWithPartialRelations,
  ListWithPartialRelations,
  NeuronWithPartialRelations,
} from 'prisma/generated/zod';
import { useEffect, useState } from 'react';

export default function ListsPane({
  currentNeuron,
  reloadNeuron,
}: {
  currentNeuron: NeuronWithPartialRelations | undefined;
  reloadNeuron: () => void;
}) {
  const { setSignInModalOpen, user, refreshUser } = useGlobalContext();
  const [addListDialogOpen, setAddListDialogOpen] = useState(false);
  const [newListDialogOpen, setNewListDialogOpen] = useState(false);
  const [addListDialogList, setAddListDialogList] = useState<ListWithPartialRelations>();
  const session = useSession();

  // https://github.com/radix-ui/primitives/issues/1241
  // https://github.com/radix-ui/primitives/discussions/1234
  useEffect(() => {
    if (!addListDialogOpen) {
      setTimeout(() => {
        document.body.style.pointerEvents = '';
      }, 500);
    }
  }, [addListDialogOpen, newListDialogOpen]);

  function listContainsNeuron(list: ListWithPartialRelations) {
    return (
      currentNeuron && currentNeuron.lists && currentNeuron?.lists?.filter((lon) => lon.listId === list.id).length > 0
    );
  }

  async function addToList(listId: string, neuron: NeuronWithPartialRelations, description: string) {
    const response = await fetch(`/api/list/add-features`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        listId,
        featuresToAdd: [
          {
            modelId: neuron.modelId,
            layer: neuron.layer,
            index: neuron.index,
            description,
          },
        ],
      }),
    });
    const results = (await response.json()) as ListsOnNeuronsWithPartialRelations[];
    return results;
  }

  async function removeFromList(listId: string, neuron: NeuronWithPartialRelations) {
    const response = await fetch(`/api/list/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        listId,
        modelId: neuron.modelId,
        layer: neuron.layer,
        index: neuron.index,
      }),
    });
    const results = (await response.json()) as ListsOnNeuronsWithPartialRelations;
    return results;
  }

  return (
    <div className="relative mb-0 mt-2 hidden flex-col rounded-lg border border-slate-200 bg-white text-xs shadow transition-all sm:mt-3 sm:flex">
      {currentNeuron && currentNeuron.lists && currentNeuron?.lists?.length > 0 ? (
        <div className="flex flex-col gap-y-1 px-5 pb-4 pt-3">
          {currentNeuron?.lists?.map((list) => (
            <div key={list.listId} className="flex flex-col">
              <div className="mt-1 flex flex-row items-center gap-x-1 text-xs leading-none text-slate-400">
                &bull;
                <a href={`/list/${list.listId}`} target="_blank" rel="noreferrer" className="text-sky-700">
                  {list.list?.name}
                </a>
              </div>
              {list.description && (
                <div className="pl-2.5 text-[10px] font-medium text-slate-400">{list.description}</div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex h-10 flex-col items-center justify-center text-xs font-bold text-slate-300 sm:h-16 sm:text-sm">
          Not in Any Lists
        </div>
      )}
      <Dialog.Root open={addListDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed left-0 top-0 h-full w-full bg-white/50">
            <Dialog.Content className="fixed left-[50%] top-[50%] max-h-[85vh] w-[90vw] max-w-[450px] translate-x-[-50%] translate-y-[-50%] rounded-md border bg-white p-[25px] text-slate-500 shadow-md focus:outline-none">
              <Dialog.Title className="mb-2 text-center text-sm font-medium text-slate-500">
                Add Feature to &#34;{addListDialogList?.name}&#34;
              </Dialog.Title>
              <Formik
                initialValues={{
                  description: '',
                }}
                onSubmit={async (values) => {
                  if (currentNeuron) {
                    await addToList(addListDialogList?.id || '', currentNeuron, values.description);
                    setAddListDialogOpen(false);
                    setAddListDialogList(undefined);
                    reloadNeuron();
                  }
                }}
              >
                {({ submitForm, isSubmitting }) => (
                  <Form
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        submitForm();
                      }
                    }}
                    className="flex max-w-sm flex-1 flex-col gap-y-1.5"
                  >
                    <Field
                      type="text"
                      name="description"
                      required
                      placeholder="(optional) Override the explanation with your own"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />

                    <div className="mt-3 flex flex-row gap-x-1">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          setAddListDialogOpen(false);
                          setAddListDialogList(undefined);
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-slate-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Cancel
                      </button>

                      <button
                        type="submit"
                        disabled={isSubmitting}
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          submitForm();
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-sky-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Add
                      </button>
                    </div>
                  </Form>
                )}
              </Formik>
            </Dialog.Content>
          </Dialog.Overlay>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={newListDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed left-0 top-0 h-full w-full bg-white/50">
            <Dialog.Content className="fixed left-[50%] top-[50%] max-h-[85vh] w-[90vw] max-w-[450px] translate-x-[-50%] translate-y-[-50%] rounded-md border bg-white p-[25px] text-slate-500 shadow-md focus:outline-none">
              <Dialog.Title className="mb-2 text-center text-sm font-medium text-slate-500">
                Create New List and Add Feature
              </Dialog.Title>
              <Formik
                initialValues={{
                  listName: '',
                  listDescription: '',
                  description: '',
                }}
                onSubmit={async (values, { resetForm }) => {
                  if (currentNeuron) {
                    if (values.listName.trim().length === 0) {
                      alert('List name cannot be blank.');
                      return;
                    }
                    const response = await fetch(`/api/list/new`, {
                      method: 'POST',
                      headers: {
                        'Content-Type': 'application/json',
                      },
                      body: JSON.stringify({
                        name: values.listName,
                        description: values.listDescription,
                      }),
                    });
                    const responseJson = (await response.json()) as ListWithPartialRelations;

                    await addToList(responseJson.id, currentNeuron, values.description);
                    resetForm();

                    setNewListDialogOpen(false);
                    reloadNeuron();
                    refreshUser();
                  }
                }}
              >
                {({ submitForm, isSubmitting }) => (
                  <Form
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        submitForm();
                      }
                    }}
                    className="flex max-w-sm flex-1 flex-col gap-y-1.5"
                  >
                    <Field
                      type="text"
                      name="listName"
                      required
                      placeholder="List Name"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />
                    <Field
                      type="text"
                      name="listDescription"
                      placeholder="List Description (optional)"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />
                    <Field
                      type="text"
                      name="description"
                      placeholder="How is this neuron relevant? (optional)"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />

                    <div className="mt-3 flex flex-row gap-x-1">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          setNewListDialogOpen(false);
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-slate-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Cancel
                      </button>

                      <button
                        type="submit"
                        disabled={isSubmitting}
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          submitForm();
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-sky-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Add
                      </button>
                    </div>
                  </Form>
                )}
              </Formik>
            </Dialog.Content>
          </Dialog.Overlay>
        </Dialog.Portal>
      </Dialog.Root>
      <DropdownMenu.Root
        onOpenChange={() => {
          // TODO: Bug - logged out currently doesn't show the sign in modal
          if (!session) {
            setSignInModalOpen(true);
            return false;
          }
          return true;
        }}
      >
        <DropdownMenu.Trigger className="flex w-full flex-1 flex-row items-center justify-center overflow-hidden rounded-b border-t bg-white px-4 py-3 text-xs font-medium leading-tight text-slate-500 outline-none hover:bg-slate-100">
          Add to List
          <Select.Icon className="ml-1 text-[10px]" />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            side="right"
            className="max-h-[340px] cursor-pointer overflow-y-auto rounded-md border border-slate-200 bg-white text-[10px] font-medium text-sky-700 shadow"
            sideOffset={-120}
          >
            {user &&
              user.lists &&
              user?.lists?.map((list) => (
                <DropdownMenu.CheckboxItem
                  key={list.id}
                  className="flex h-8 min-w-[200px] flex-row items-center overflow-hidden border-b px-3 font-sans text-xs hover:bg-slate-100 focus:outline-none"
                  checked={listContainsNeuron(list as ListWithPartialRelations)}
                  onSelect={async () => {
                    if (currentNeuron && list.id) {
                      if (listContainsNeuron(list as ListWithPartialRelations)) {
                        if (window.confirm(`Are you sure you want to remove this neuron from "${list.name}"?`)) {
                          await removeFromList(list.id, currentNeuron);
                          reloadNeuron();
                        }
                      } else {
                        setAddListDialogList(list as ListWithPartialRelations);
                        setAddListDialogOpen(true);
                      }
                    }
                  }}
                >
                  <div className="w-7">
                    <DropdownMenu.ItemIndicator>
                      <Check className="h-4 w-4" />
                    </DropdownMenu.ItemIndicator>
                  </div>
                  <div className="flex items-center justify-center leading-none">{list.name}</div>
                </DropdownMenu.CheckboxItem>
              ))}
            <DropdownMenu.CheckboxItem
              className="sticky bottom-0 flex h-8 min-w-[200px] flex-row items-center overflow-hidden border-b bg-slate-100 px-3 font-sans text-xs hover:bg-slate-200 focus:outline-none"
              onSelect={() => {
                setNewListDialogOpen(true);
              }}
            >
              <div className="w-7">
                <DropdownMenu.ItemIndicator>
                  <Check className="h-4 w-4" />
                </DropdownMenu.ItemIndicator>
              </div>
              <div className="flex items-center justify-center leading-none">+ New List</div>
            </DropdownMenu.CheckboxItem>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  );
}
