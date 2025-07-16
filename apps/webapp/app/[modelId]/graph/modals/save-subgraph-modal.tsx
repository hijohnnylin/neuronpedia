'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import { GraphMetadataSubgraph, GraphMetadataSubgraphWithPartialRelations } from '@/prisma/generated/zod';
import copy from 'copy-to-clipboard';
import { Field, Form, Formik } from 'formik';
import { CopyIcon } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useEffect, useState } from 'react';
import { SaveSubgraphRequest } from '../utils';
import SubgraphItem from './subgraph-item';

export default function SaveSubgraphModal() {
  const { isSaveSubgraphModalOpen, setIsSaveSubgraphModalOpen } = useGraphModalContext();
  const { visState, selectedMetadataGraph, selectedModelId } = useGraphContext();
  const session = useSession();
  const { setSignInModalOpen } = useGlobalContext();
  const [savedSubgraph, setSavedSubgraph] = useState<{ subgraphId: string; modelId: string; slug: string } | null>(
    null,
  );
  const [existingSubgraphs, setExistingSubgraphs] = useState<GraphMetadataSubgraphWithPartialRelations[]>([]);
  const [loadingSubgraphs, setLoadingSubgraphs] = useState(false);

  // Fetch existing subgraphs when modal opens and user is logged in
  useEffect(() => {
    if (isSaveSubgraphModalOpen && session.data?.user && selectedModelId && selectedMetadataGraph?.slug) {
      const fetchExistingSubgraphs = async () => {
        setLoadingSubgraphs(true);
        try {
          const response = await fetch('/api/graph/subgraph/list', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              modelId: selectedModelId,
              slug: selectedMetadataGraph.slug,
            }),
          });
          if (response.ok) {
            const data = await response.json();
            setExistingSubgraphs(data.subgraphs || []);
          }
        } catch (error) {
          console.error('Error fetching existing subgraphs:', error);
        } finally {
          setLoadingSubgraphs(false);
        }
      };
      fetchExistingSubgraphs();
    }
  }, [isSaveSubgraphModalOpen, session.data?.user, selectedModelId, selectedMetadataGraph?.slug]);

  const saveSubgraph = async (displayName?: string, overwriteId?: string) => {
    try {
      const modelId = selectedModelId;
      const slug = selectedMetadataGraph?.slug;
      if (!slug) {
        alert('No slug found for selected graph');
        return null;
      }

      const body: SaveSubgraphRequest = {
        modelId,
        slug,
        displayName,
        pinnedIds: visState.pinnedIds,
        supernodes: visState.subgraph?.supernodes || [],
        clerps: visState.clerps,
        pruningThreshold: visState.pruningThreshold ?? null,
        densityThreshold: visState.densityThreshold ?? null,
      };

      if (overwriteId) {
        body.overwriteId = overwriteId;
      }

      const response = await fetch(`/api/graph/subgraph/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Failed to ${overwriteId ? 'overwrite' : 'save'} subgraph`);
      }

      const data = await response.json();
      console.log(`Subgraph ${overwriteId ? 'overwritten' : 'saved'} with id:`, data.subgraphId);

      // Set saved subgraph data to show success view
      setSavedSubgraph({
        subgraphId: data.subgraphId,
        modelId,
        slug,
      });

      return data;
    } catch (error) {
      console.error(`Error ${overwriteId ? 'overwriting' : 'saving'} subgraph:`, error);
      alert(`Failed to ${overwriteId ? 'overwrite' : 'save'} subgraph: ${error}`);
      return null;
    }
  };

  const handleOverwriteSubgraph = async (subgraph: GraphMetadataSubgraph) => {
    // Use window.confirm but handle it properly for the linter
    const shouldOverwrite = window.confirm(
      `Are you sure you want to overwrite "${subgraph.displayName || `${session.data?.user.name}'s Subgraph`}"?\nThis action cannot be undone.`,
    );
    if (!shouldOverwrite) {
      return;
    }

    await saveSubgraph(subgraph.displayName || undefined, subgraph.id);
  };

  // Check if user is not logged in
  if (!session.data?.user) {
    return (
      <Dialog open={isSaveSubgraphModalOpen} onOpenChange={setIsSaveSubgraphModalOpen}>
        <DialogContent className="bg-white text-slate-600 sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Save Subgraph</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-slate-600">Signing in is required to save subgraphs to your account.</p>
            <div className="flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => setIsSaveSubgraphModalOpen(false)}
                className="rounded-md border border-slate-300 bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 focus:outline-none focus:ring-0 focus:ring-slate-500"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsSaveSubgraphModalOpen(false);
                  setSignInModalOpen(true);
                }}
                className="rounded-md border border-transparent bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700 focus:outline-none focus:ring-0 focus:ring-sky-500"
              >
                Sign In
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // Show success view if subgraph was saved
  if (savedSubgraph) {
    const subgraphUrl = `${NEXT_PUBLIC_URL}/${savedSubgraph.modelId}/graph?slug=${savedSubgraph.slug}&subgraph=${savedSubgraph.subgraphId}`;

    return (
      <Dialog
        open={isSaveSubgraphModalOpen}
        onOpenChange={() => {
          setIsSaveSubgraphModalOpen(false);
          setSavedSubgraph(null);
        }}
      >
        <DialogContent className="bg-white text-slate-600 sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Save Subgraph</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="text-center">
              <p className="mb-2 text-left text-sm font-semibold text-emerald-600">Subgraph Saved</p>
              <div className="space-y-2">
                <div className="block text-left text-xs font-medium text-slate-600">Copy Link for Sharing</div>
                <div className="flex flex-row items-center justify-center gap-x-1.5">
                  <input
                    type="text"
                    disabled
                    className="disabled form-input block w-full rounded-md border-2 border-slate-300 bg-slate-200 text-xs text-slate-500 focus:outline-none focus:ring-0"
                    value={subgraphUrl}
                  />
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      copy(subgraphUrl);
                      alert('Copied to clipboard.');
                    }}
                  >
                    <CopyIcon className="h-9 min-w-12 rounded-md bg-slate-200 px-2.5 py-2.5 text-slate-500 hover:bg-slate-300 active:bg-sky-300" />
                  </button>
                </div>
              </div>
            </div>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => {
                  setIsSaveSubgraphModalOpen(false);
                  setSavedSubgraph(null);
                }}
                className="rounded-md border border-transparent bg-sky-600 px-4 py-1 text-sm font-medium text-white hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-500"
              >
                Done
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={isSaveSubgraphModalOpen} onOpenChange={setIsSaveSubgraphModalOpen}>
      <DialogContent className="max-h-[80vh] overflow-y-auto bg-white text-slate-600 sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Save Subgraph</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <h3 className="mb-2 text-sm font-bold text-slate-600">Overwrite Existing</h3>
          {/* Overwrite Existing Subgraph Section */}
          {loadingSubgraphs ? (
            <div className="flex flex-col items-center justify-center gap-y-3 py-4">
              <LoadingSquare className="h-4 w-4" />
              <p className="text-xs text-slate-500">Loading your subgraphs...</p>
            </div>
          ) : existingSubgraphs.length > 0 ? (
            <div className="border-b border-slate-200 pb-4">
              <div className="ml-3 max-h-64 space-y-0 overflow-y-auto rounded-md border border-slate-200">
                {existingSubgraphs.map((subgraph) => (
                  <SubgraphItem
                    key={subgraph.id}
                    subgraph={subgraph}
                    onClick={handleOverwriteSubgraph}
                    onDelete={() => {
                      setExistingSubgraphs(existingSubgraphs.filter((s) => s.id !== subgraph.id));
                    }}
                    fallbackDisplayName={`${session.data?.user.name}'s Subgraph`}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="py-4 text-center">
              <p className="text-xs font-medium text-slate-400">You have no saved subgraphs for this graph.</p>
            </div>
          )}

          {/* Save New Subgraph Section */}
          <div>
            <h3 className="mb-1.5 text-sm font-bold text-slate-600">New Subgraph</h3>
            <Formik
              initialValues={{
                name: '',
              }}
              onSubmit={async (values, { setSubmitting }) => {
                const result = await saveSubgraph(values.name);
                if (result) {
                  // Success is already handled in saveSubgraph
                }
                setSubmitting(false);
              }}
            >
              {({ isSubmitting }) => (
                <Form className="ml-3 space-y-2">
                  <div>
                    <div className="mb-1 flex items-center justify-start gap-x-1 text-xs font-medium text-slate-600">
                      Optional Name{' '}
                      <span className="text-xs text-slate-400">
                        - Defaults to &quot;{session.data?.user.name}&apos;s Subgraph&quot;
                      </span>
                    </div>

                    <Field
                      type="text"
                      name="name"
                      id="name"
                      maxLength={32}
                      className="w-full rounded-md border border-slate-300 px-3 py-2 text-xs shadow-sm placeholder:text-slate-300 focus:outline-none"
                      placeholder="Enter a name for this subgraph"
                    />
                  </div>
                  <div className="flex justify-end space-x-2">
                    <button
                      type="button"
                      onClick={() => setIsSaveSubgraphModalOpen(false)}
                      className="rounded-md border border-slate-300 bg-slate-100 px-4 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200 focus:outline-none focus:ring-2 focus:ring-slate-500"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="rounded-md border border-transparent bg-sky-600 px-4 py-1 text-xs font-medium text-white hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isSubmitting ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </Form>
              )}
            </Formik>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
