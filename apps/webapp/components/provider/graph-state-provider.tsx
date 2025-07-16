'use client';

import { CLTGraphNode } from '@/app/[modelId]/graph/utils';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { ReactNode, createContext, useCallback, useContext, useMemo, useRef } from 'react';

// Define the graph state context type
type GraphStateContextType = {
  // Hover state
  hoveredIdRef: React.MutableRefObject<string | null>;
  hoveredCtxIdxRef: React.MutableRefObject<number | null>;
  updateHoverState: (node: CLTGraphNode | null, onHoverChange?: (hoveredId: string | null) => void) => void;
  clearHoverState: (onHoverChange?: (hoveredId: string | null) => void) => void;

  // Clicked state
  clickedIdRef: React.MutableRefObject<string | null>;
  clickedCtxIdxRef: React.MutableRefObject<number | null>;
  updateClickedState: (node: CLTGraphNode | null, onClickedChange?: (clickedId: string | null) => void) => void;
  clearClickedState: (onClickedChange?: (clickedId: string | null) => void) => void;

  // Callback registration for bidirectional hover
  registerHoverCallback: (callback: (hoveredId: string | null) => void) => () => void;

  // Callback registration for bidirectional clicked
  registerClickedCallback: (callback: (clickedId: string | null) => void) => () => void;
};

// Create the context
const GraphStateContext = createContext<GraphStateContextType | undefined>(undefined);

// Provider component
export function GraphStateProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const hoveredIdRef = useRef<string | null>(null);
  const hoveredCtxIdxRef = useRef<number | null>(null);
  const clickedIdRef = useRef<string | null>(null);
  const clickedCtxIdxRef = useRef<number | null>(null);

  // Store callbacks for hover changes
  const hoverCallbacksRef = useRef<Set<(hoveredId: string | null) => void>>(new Set());

  // Store callbacks for clicked changes
  const clickedCallbacksRef = useRef<Set<(clickedId: string | null) => void>>(new Set());

  const updateUrlParams = useCallback(
    (keysToValues: Record<string, string | null>) => {
      const params = new URLSearchParams(searchParams.toString());
      Object.entries(keysToValues).forEach(([key, value]) => {
        if (value) {
          params.set(key, value);
        } else {
          params.delete(key);
        }
      });
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  const registerHoverCallback = useCallback((callback: (hoveredId: string | null) => void) => {
    hoverCallbacksRef.current.add(callback);

    // Return cleanup function
    return () => {
      hoverCallbacksRef.current.delete(callback);
    };
  }, []);

  const registerClickedCallback = useCallback((callback: (clickedId: string | null) => void) => {
    clickedCallbacksRef.current.add(callback);

    // Return cleanup function
    return () => {
      clickedCallbacksRef.current.delete(callback);
    };
  }, []);

  const updateHoverState = useCallback(
    (node: CLTGraphNode | null, onHoverChange?: (hoveredId: string | null) => void) => {
      const newHoveredId = node?.featureId || null;
      const newHoveredCtxIdx = node?.ctx_idx || null;

      // don't update if window width is less than 640px
      if (window.innerWidth < 640) {
        return;
      }

      // Only update if the value actually changed
      if (hoveredIdRef.current !== newHoveredId) {
        hoveredIdRef.current = newHoveredId;
        hoveredCtxIdxRef.current = newHoveredCtxIdx;

        // Trigger the D3 callback for link-graph
        if (onHoverChange) {
          onHoverChange(newHoveredId);
        }

        // Notify all registered callbacks for bidirectional updates
        hoverCallbacksRef.current.forEach((callback) => callback(newHoveredId));
      }
    },
    [],
  );

  const clearHoverState = useCallback((onHoverChange?: (hoveredId: string | null) => void) => {
    if (hoveredIdRef.current !== null) {
      hoveredIdRef.current = null;
      hoveredCtxIdxRef.current = null;

      // Trigger the D3 callback for link-graph
      if (onHoverChange) {
        onHoverChange(null);
      }

      // Notify all registered callbacks for bidirectional updates
      hoverCallbacksRef.current.forEach((callback) => callback(null));
    }
  }, []);

  const updateClickedState = useCallback(
    (node: CLTGraphNode | null, onClickedChange?: (clickedId: string | null) => void) => {
      const newClickedId = node?.nodeId || null;
      const newClickedCtxIdx = node?.ctx_idx || null;

      // Only update if the value actually changed
      if (clickedIdRef.current !== newClickedId) {
        clickedIdRef.current = newClickedId;
        clickedCtxIdxRef.current = newClickedCtxIdx;

        // // Update URL parameter
        // updateUrlParams({ clickedId: newClickedId });

        // Trigger the D3 callback for link-graph
        if (onClickedChange) {
          onClickedChange(newClickedId);
        }
        // Notify all registered callbacks for bidirectional updates
        clickedCallbacksRef.current.forEach((callback) => callback(newClickedId));
      }
    },
    [updateUrlParams],
  );

  const clearClickedState = useCallback((onClickedChange?: (clickedId: string | null) => void) => {
    if (clickedIdRef.current !== null) {
      clickedIdRef.current = null;
      clickedCtxIdxRef.current = null;

      // Trigger the D3 callback for link-graph
      if (onClickedChange) {
        onClickedChange(null);
      }

      // Notify all registered callbacks for bidirectional updates
      clickedCallbacksRef.current.forEach((callback) => callback(null));
    }
  }, []);

  const value = useMemo(
    () => ({
      hoveredIdRef,
      hoveredCtxIdxRef,
      updateHoverState,
      clearHoverState,
      clickedIdRef,
      clickedCtxIdxRef,
      updateClickedState,
      clearClickedState,
      registerHoverCallback,
      registerClickedCallback,
    }),
    [
      hoveredIdRef,
      hoveredCtxIdxRef,
      updateHoverState,
      clearHoverState,
      clickedIdRef,
      clickedCtxIdxRef,
      updateClickedState,
      clearClickedState,
      registerHoverCallback,
      registerClickedCallback,
    ],
  );

  return <GraphStateContext.Provider value={value}>{children}</GraphStateContext.Provider>;
}

// Hook to use the graph state context
export function useGraphStateContext() {
  const context = useContext(GraphStateContext);
  if (context === undefined) {
    throw new Error('useGraphStateContext must be used within a GraphStateProvider');
  }
  return context;
}
