'use client';

import { createContext, ReactNode, useContext, useMemo, useState } from 'react';

type AssistantAxisModalContextType = {
    isWelcomeModalOpen: boolean;
    setIsWelcomeModalOpen: (isOpen: boolean) => void;
};

const AssistantAxisModalContext = createContext<AssistantAxisModalContextType | undefined>(undefined);

export function AssistantAxisModalProvider({ children }: { children: ReactNode }) {
    const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState<boolean>(false);

    const contextValue = useMemo(
        () => ({
            isWelcomeModalOpen,
            setIsWelcomeModalOpen,
        }),
        [isWelcomeModalOpen],
    );

    return <AssistantAxisModalContext.Provider value={contextValue}>{children}</AssistantAxisModalContext.Provider>;
}

export function useAssistantAxisModalContext() {
    const context = useContext(AssistantAxisModalContext);
    if (context === undefined) {
        throw new Error('useAssistantAxisModalContext must be used within an AssistantAxisModalProvider');
    }
    return context;
}

