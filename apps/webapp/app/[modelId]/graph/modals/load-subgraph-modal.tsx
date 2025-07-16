'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import { GraphMetadataSubgraphWithPartialRelations } from '@/prisma/generated/zod';
import { useSession } from 'next-auth/react';
import { useEffect, useState } from 'react';
import SubgraphItem from './subgraph-item';

export default function LoadSubgraphModal() {
  const { isLoadSubgraphModalOpen, setIsLoadSubgraphModalOpen } = useGraphModalContext();
  const { selectedMetadataGraph, selectedModelId, loadSubgraph } = useGraphContext();
  const { clearClickedState, clearHoverState } = useGraphStateContext();
  const session = useSession();
  const [existingSubgraphs, setExistingSubgraphs] = useState<GraphMetadataSubgraphWithPartialRelations[]>([]);
  const [featuredSubgraphs, setFeaturedSubgraphs] = useState<GraphMetadataSubgraphWithPartialRelations[]>([]);
  const [loadingSubgraphs, setLoadingSubgraphs] = useState(false);
  const [loadingFeatured, setLoadingFeatured] = useState(false);
  const { setSignInModalOpen } = useGlobalContext();

  // Fetch featured subgraphs when modal opens
  useEffect(() => {
    if (isLoadSubgraphModalOpen && selectedModelId && selectedMetadataGraph?.slug) {
      const fetchFeaturedSubgraphs = async () => {
        setLoadingFeatured(true);
        try {
          const response = await fetch('/api/graph/subgraph/list-featured', {
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
            setFeaturedSubgraphs(data.subgraphs || []);
          }
        } catch (error) {
          console.error('Error fetching featured subgraphs:', error);
        } finally {
          setLoadingFeatured(false);
        }
      };
      fetchFeaturedSubgraphs();
    }
  }, [isLoadSubgraphModalOpen, selectedModelId, selectedMetadataGraph?.slug]);

  // Fetch existing subgraphs when modal opens and user is logged in
  useEffect(() => {
    if (isLoadSubgraphModalOpen && session.data?.user && selectedModelId && selectedMetadataGraph?.slug) {
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
  }, [isLoadSubgraphModalOpen, session.data?.user, selectedModelId, selectedMetadataGraph?.slug]);

  const handleLoadSubgraph = async (subgraph: GraphMetadataSubgraphWithPartialRelations) => {
    loadSubgraph(subgraph);
    clearClickedState();
    clearHoverState();
    // Close the modal after clicking
    setIsLoadSubgraphModalOpen(false);
  };

  return (
    <Dialog open={isLoadSubgraphModalOpen} onOpenChange={setIsLoadSubgraphModalOpen}>
      <DialogContent className="max-h-[80vh] overflow-y-auto bg-white text-slate-600 sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Load Subgraph</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <h3 className="mb-2 text-sm font-bold text-slate-600">Featured Subgraphs</h3>
          {/* Featured Subgraphs Section */}
          {loadingFeatured ? (
            <div className="flex flex-col items-center justify-center gap-y-3 py-4">
              <LoadingSquare className="h-4 w-4" />
              <p className="text-xs text-slate-500">Loading featured subgraphs...</p>
            </div>
          ) : featuredSubgraphs.length > 0 ? (
            <div className="border-b border-slate-200 pb-4">
              <div className="ml-3 max-h-64 space-y-0 overflow-y-auto rounded-md border border-slate-200">
                {featuredSubgraphs.map((subgraph) => (
                  <SubgraphItem
                    key={subgraph.id}
                    subgraph={subgraph}
                    onClick={handleLoadSubgraph}
                    onDelete={() => {
                      setFeaturedSubgraphs(featuredSubgraphs.filter((s) => s.id !== subgraph.id));
                    }}
                    fallbackDisplayName="Featured Subgraph"
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="py-4 text-center">
              <p className="text-xs font-medium text-slate-400">
                There are no featured subgraphs/solutions for this graph.
              </p>
            </div>
          )}

          {/* Load User Subgraph Section */}

          <h3 className="mb-2 text-sm font-bold text-slate-600">Your Saved Subgraphs</h3>
          {loadingSubgraphs ? (
            <div className="flex flex-col items-center justify-center gap-y-3 py-4">
              <LoadingSquare className="h-4 w-4" />
              <p className="text-xs text-slate-500">Loading your subgraphs...</p>
            </div>
          ) : existingSubgraphs.length > 0 ? (
            <div>
              <div className="ml-3 max-h-64 space-y-0 overflow-y-auto rounded-md border border-slate-200">
                {existingSubgraphs.map((subgraph) => (
                  <SubgraphItem
                    key={subgraph.id}
                    subgraph={subgraph}
                    onClick={handleLoadSubgraph}
                    onDelete={() => {
                      setExistingSubgraphs(existingSubgraphs.filter((s) => s.id !== subgraph.id));
                    }}
                    fallbackDisplayName={`${session.data?.user.name}'s Subgraph`}
                  />
                ))}
              </div>
            </div>
          ) : !session.data?.user ? (
            <div className="pb-2 text-center">
              <p className="ml-0 text-xs text-slate-500">Loading your saved graphs requires signing in.</p>
              <button
                type="button"
                onClick={() => {
                  setIsLoadSubgraphModalOpen(false);
                  setSignInModalOpen(true);
                }}
                className="ml-0 mt-1.5 rounded-md border border-transparent bg-sky-600 px-4 py-1 text-xs font-medium text-white hover:bg-sky-700 focus:outline-none"
              >
                Sign In
              </button>
            </div>
          ) : (
            <div className="py-4 text-center">
              <p className="text-xs font-medium text-slate-400">You have no saved subgraphs for this graph.</p>
            </div>
          )}

          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setIsLoadSubgraphModalOpen(false)}
              className="rounded-md bg-slate-100 px-4 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200 focus:outline-none"
            >
              Cancel
            </button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
