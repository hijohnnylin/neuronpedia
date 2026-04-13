'use client';

import { useRouter } from '@bprogress/next';
import * as Dialog from '@radix-ui/react-dialog';
import { Field, Form, Formik } from 'formik';
import { ListWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';

export default function Lists({
  showCreate,
  initialLists,
  username,
}: {
  showCreate: boolean;
  initialLists: ListWithPartialRelations[];
  username: string;
}) {
  const [lists, setLists] = useState<ListWithPartialRelations[]>([]);
  const [newListDialogOpen, setNewListDialogOpen] = useState(false);

  useEffect(() => {
    setLists(initialLists);
  }, [setLists, initialLists]);

  const router = useRouter();

  return (
    <div className="flex w-full max-w-screen-lg flex-col items-center pt-5">
      <div className="flex w-full max-w-screen-lg flex-row justify-between">
        <h1 className="mb-0 px-3 py-1 text-left text-base font-medium text-slate-500 xl:px-0">Lists by {username}</h1>
        {showCreate && (
          <button
            type="button"
            className="rounded bg-sky-700 px-3 py-2 text-xs text-white"
            onClick={() => {
              setNewListDialogOpen(true);
            }}
          >
            + New List
          </button>
        )}
      </div>

      <div className="mt-3 grid w-full max-w-screen-lg grid-cols-1 items-start justify-center gap-x-2 gap-y-2 px-3 pb-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:px-0">
        {lists.length === 0 && (
          <div className="col-span-1 mt-5 text-center text-sm text-slate-400 sm:col-span-full">
            No lists found. Create one with + New List above.
          </div>
        )}
        {lists.map((list) => (
          <button
            type="button"
            key={list.id}
            onClick={() => {
              router.push(`/list/${list.id}`);
            }}
            className="col-span-1 flex cursor-pointer flex-col gap-y-0.5 rounded border border-slate-200 bg-white px-4 py-3 text-slate-600 hover:bg-slate-200/30"
          >
            <div className="text-sm font-medium">{list.name}</div>
            <div className="mb-1 mt-1 line-clamp-3 text-xs text-slate-400">{list.description}</div>
            <div className="text-[11px] font-medium text-slate-400">Created on {list.createdAt.toLocaleString()}</div>
          </button>
        ))}
        <Dialog.Root open={newListDialogOpen}>
          <Dialog.Portal>
            <Dialog.Overlay className="fixed left-0 top-0 h-full w-full bg-white/50">
              <Dialog.Content className="fixed left-[50%] top-[50%] max-h-[85vh] w-[90vw] max-w-[450px] translate-x-[-50%] translate-y-[-50%] rounded-md border bg-white p-[25px] text-slate-500 shadow-md focus:outline-none">
                <Dialog.Title className="mb-2 text-center text-sm font-medium text-slate-500">
                  Create New List
                </Dialog.Title>
                <Formik
                  initialValues={{
                    listName: '',
                    listDescription: '',
                    description: '',
                  }}
                  onSubmit={async (values, { resetForm }) => {
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
                    const results = (await response.json()) as ListWithPartialRelations;
                    setLists([...lists, results]);
                    resetForm();

                    setNewListDialogOpen(false);
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
      </div>
    </div>
  );
}
