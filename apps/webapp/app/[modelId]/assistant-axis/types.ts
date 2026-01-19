export interface PersonaCheckTurn {
  pc_values: Record<string, number>;
  snippet: string;
}

export interface PersonaCheckResult {
  pc_titles: string[];
  turns: PersonaCheckTurn[];
}

// Assistant axis data with steer type indicator
export interface AssistantAxisItem {
  type: 'DEFAULT' | 'STEERED';
  pc_titles: string[];
  turns: PersonaCheckTurn[];
}

