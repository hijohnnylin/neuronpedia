import { createContext, ReactNode, useContext, useMemo, useState } from 'react';

// Define the modal context type
type GraphModalContextType = {
  // Copy modal state
  isCopyModalOpen: boolean;
  setIsCopyModalOpen: (isOpen: boolean) => void;

  // Welcome modal state
  isWelcomeModalOpen: boolean;
  setIsWelcomeModalOpen: (isOpen: boolean) => void;
  welcomeModalInitialStep: number | null;
  openWelcomeModalToStep: (step: number) => void;
  resetWelcomeModalStep: () => void;

  // Generate graph modal state
  isGenerateGraphModalOpen: boolean;
  setIsGenerateGraphModalOpen: (isOpen: boolean) => void;

  // Save subgraph modal state
  isSaveSubgraphModalOpen: boolean;
  setIsSaveSubgraphModalOpen: (isOpen: boolean) => void;

  // Load subgraph modal state
  isLoadSubgraphModalOpen: boolean;
  setIsLoadSubgraphModalOpen: (isOpen: boolean) => void;

  // Steer modal state
  isSteerModalOpen: boolean;
  setIsSteerModalOpen: (isOpen: boolean) => void;
};

// Create the modal context
const GraphModalContext = createContext<GraphModalContextType | undefined>(undefined);

// GraphModalProvider component
export function GraphModalProvider({ children }: { children: ReactNode }) {
  const [isCopyModalOpen, setIsCopyModalOpenState] = useState<boolean>(false);
  const [isWelcomeModalOpen, setIsWelcomeModalOpenState] = useState<boolean>(false);
  const [welcomeModalInitialStep, setWelcomeModalInitialStep] = useState<number | null>(null);
  const [isGenerateGraphModalOpen, setIsGenerateGraphModalOpenState] = useState<boolean>(false);
  const [isSaveSubgraphModalOpen, setIsSaveSubgraphModalOpenState] = useState<boolean>(false);
  const [isLoadSubgraphModalOpen, setIsLoadSubgraphModalOpenState] = useState<boolean>(false);
  const [isSteerModalOpen, setIsSteerModalOpenState] = useState<boolean>(false);

  // Custom setter for copy modal that closes other modals
  const setIsCopyModalOpen = (isOpen: boolean) => {
    setIsCopyModalOpenState(isOpen);
    if (isOpen) {
      setIsWelcomeModalOpenState(false);
      setIsGenerateGraphModalOpenState(false);
      setIsSaveSubgraphModalOpenState(false);
      setIsLoadSubgraphModalOpenState(false);
      setIsSteerModalOpenState(false);
      setWelcomeModalInitialStep(null);
    }
  };

  // Custom setter for welcome modal that closes other modals
  const setIsWelcomeModalOpen = (isOpen: boolean) => {
    setIsWelcomeModalOpenState(isOpen);
    if (isOpen) {
      setIsCopyModalOpenState(false);
      setIsGenerateGraphModalOpenState(false);
      setIsSaveSubgraphModalOpenState(false);
      setIsLoadSubgraphModalOpenState(false);
      setIsSteerModalOpenState(false);
    } else {
      setWelcomeModalInitialStep(null);
    }
  };

  // Custom setter for generate graph modal that closes other modals
  const setIsGenerateGraphModalOpen = (isOpen: boolean) => {
    setIsGenerateGraphModalOpenState(isOpen);
    if (isOpen) {
      setIsCopyModalOpenState(false);
      setIsWelcomeModalOpenState(false);
      setWelcomeModalInitialStep(null);
      setIsSaveSubgraphModalOpenState(false);
      setIsLoadSubgraphModalOpenState(false);
      setIsSteerModalOpenState(false);
    }
  };

  // Custom setter for save subgraph modal that closes other modals
  const setIsSaveSubgraphModalOpen = (isOpen: boolean) => {
    setIsSaveSubgraphModalOpenState(isOpen);
    if (isOpen) {
      setIsCopyModalOpenState(false);
      setIsWelcomeModalOpenState(false);
      setWelcomeModalInitialStep(null);
      setIsGenerateGraphModalOpenState(false);
      setIsLoadSubgraphModalOpenState(false);
      setIsSteerModalOpenState(false);
    }
  };

  // Custom setter for load subgraph modal that closes other modals
  const setIsLoadSubgraphModalOpen = (isOpen: boolean) => {
    setIsLoadSubgraphModalOpenState(isOpen);
    if (isOpen) {
      setIsCopyModalOpenState(false);
      setIsWelcomeModalOpenState(false);
      setWelcomeModalInitialStep(null);
      setIsGenerateGraphModalOpenState(false);
      setIsSaveSubgraphModalOpenState(false);
      setIsSteerModalOpenState(false);
    }
  };

  // Custom setter for steer modal that closes other modals
  const setIsSteerModalOpen = (isOpen: boolean) => {
    setIsSteerModalOpenState(isOpen);
    if (isOpen) {
      setIsCopyModalOpenState(false);
      setIsWelcomeModalOpenState(false);
      setWelcomeModalInitialStep(null);
      setIsGenerateGraphModalOpenState(false);
      setIsSaveSubgraphModalOpenState(false);
      setIsLoadSubgraphModalOpenState(false);
    }
  };

  // Function to open welcome modal to a specific step
  const openWelcomeModalToStep = (step: number) => {
    setWelcomeModalInitialStep(step);
    setIsWelcomeModalOpenState(true);
    // Close other modals
    setIsCopyModalOpenState(false);
    setIsGenerateGraphModalOpenState(false);
    setIsSaveSubgraphModalOpenState(false);
    setIsLoadSubgraphModalOpenState(false);
    setIsSteerModalOpenState(false);
  };

  // Function to reset welcome modal initial step
  const resetWelcomeModalStep = () => {
    setWelcomeModalInitialStep(null);
  };

  const contextValue = useMemo(
    () => ({
      isCopyModalOpen,
      setIsCopyModalOpen,
      isWelcomeModalOpen,
      setIsWelcomeModalOpen,
      welcomeModalInitialStep,
      openWelcomeModalToStep,
      resetWelcomeModalStep,
      isGenerateGraphModalOpen,
      setIsGenerateGraphModalOpen,
      isSaveSubgraphModalOpen,
      setIsSaveSubgraphModalOpen,
      isLoadSubgraphModalOpen,
      setIsLoadSubgraphModalOpen,
      isSteerModalOpen,
      setIsSteerModalOpen,
    }),
    [
      isCopyModalOpen,
      isWelcomeModalOpen,
      welcomeModalInitialStep,
      isGenerateGraphModalOpen,
      isSaveSubgraphModalOpen,
      isLoadSubgraphModalOpen,
      isSteerModalOpen,
    ],
  );

  return <GraphModalContext.Provider value={contextValue}>{children}</GraphModalContext.Provider>;
}

// Custom hook to use the modal context
export function useGraphModalContext() {
  const context = useContext(GraphModalContext);
  if (context === undefined) {
    throw new Error('useGraphModalContext must be used within a GraphModalProvider');
  }
  return context;
}
