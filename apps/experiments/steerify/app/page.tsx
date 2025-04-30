"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// Types
type SearchResult = {
  modelId: string;
  layer: string;
  index: number;
  description: string;
  neuron: {
    maxActApprox: number;
  };
  steeredCompletion: string | null;
  isLoading: boolean;
};

type SteerResponse = {
  outputs: [
    {
      output: string;
    }
  ];
};

const LoadingSpinner = () => (
  <div className="flex items-center justify-center">
    <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-primary"></div>
  </div>
);

export default function Home() {
  const [inferenceServerActive, setInferenceServerActive] = useState<boolean | null>(null);
  const [steerPrompt, setSteerPrompt] = useState("I often think about");
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [complete, setComplete] = useState(false);

  // Check if inference server is running
  useEffect(() => {
    const checkInferenceServer = async () => {
      try {
        const response = await fetch("http://localhost:5002/health");
        setInferenceServerActive(response.ok);
      } catch (error) {
        setInferenceServerActive(false);
      }
    };
    
    checkInferenceServer();
  }, []);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setSearchResults([]);
    setComplete(false);
    
    try {
      const response = await fetch("/api/explanation/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modelId: "gpt2-small",
          layers: ["5-res-jb", "6-res-jb", "7-res-jb"],
          query: searchQuery,
        }),
      });
      
      if (!response.ok) {
        throw new Error("Search failed");
      }
      
      const data = await response.json();
      const top5Results = data.results.slice(0, 5).map((result: any) => ({
        modelId: result.modelId,
        layer: result.layer,
        index: parseInt(result.index, 10),
        description: result.description,
        neuron: {
          maxActApprox: result.neuron.maxActApprox,
        },
        steeredCompletion: null,
        isLoading: true,
      }));
      
      setSearchResults(top5Results);
      
      // For each result, fetch the steered completion
      top5Results.forEach((result, index) => {
        handleSteer(result, index);
      });
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSteer = async (result: SearchResult, index: number) => {
    try {
      const response = await fetch("http://localhost:5002/v1/steer/completion", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: steerPrompt,
          model: result.modelId,
          features: [
            {
              model: result.modelId,
              source: result.layer,
              index: result.index,
              strength: result.neuron.maxActApprox,
            },
          ],
          types: ["STEERED"],
          n_completion_tokens: 16,
          temperature: 0.5,
          strength_multiplier: 1.5,
          freq_penalty: 1,
          seed: 16,
          steer_method: "SIMPLE_ADDITIVE",
          normalize_steering: false,
        }),
      });
      
      if (!response.ok) {
        throw new Error("Steering failed");
      }
      
      const data: SteerResponse = await response.json();
      const steeredOutput = data.outputs[0].output;
      
      setSearchResults((prev) => {
        const newResults = [...prev];
        newResults[index] = {
          ...newResults[index],
          steeredCompletion: steeredOutput,
          isLoading: false,
        };
        
        // Check if all steerings are complete
        const allComplete = newResults.every((r) => !r.isLoading);
        if (allComplete) {
          setComplete(true);
        }
        
        return newResults;
      });
    } catch (error) {
      console.error("Steering error:", error);
      
      setSearchResults((prev) => {
        const newResults = [...prev];
        newResults[index] = {
          ...newResults[index],
          steeredCompletion: "Error fetching completion",
          isLoading: false,
        };
        return newResults;
      });
    }
  };

  const handleReset = () => {
    setSearchQuery("");
    setSearchResults([]);
    setComplete(false);
  };

  const exampleQueries = ["dogs", "happiness", "star wars"];

  // If inference server is not active, show error message
  if (inferenceServerActive === false) {
    return (
      <div className="flex flex-col items-center justify-center space-y-6 p-8 text-center">
        <div className="text-destructive text-xl font-semibold">
          Inference Server Not Running
        </div>
        <p>
          Please start the inference server with the following command:
        </p>
        <pre className="bg-muted rounded-md p-4 text-sm overflow-x-auto">
          cd apps/inference && poetry install && poetry run python start.py
        </pre>
        <Button onClick={() => setInferenceServerActive(null)}>Retry Connection</Button>
      </div>
    );
  }

  return (
    <div className="container mx-auto space-y-8">
      <div className="space-y-4">
        <div>
          <label htmlFor="steerPrompt" className="text-sm font-medium">
            Steer Prompt
          </label>
          <Input
            id="steerPrompt"
            value={steerPrompt}
            onChange={(e) => setSteerPrompt(e.target.value)}
            className="mt-1"
          />
        </div>

        <div>
          <label htmlFor="searchQuery" className="text-sm font-medium">
            Search Query
          </label>
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((query) => (
                <Button
                  key={query}
                  variant="outline"
                  size="sm"
                  onClick={() => setSearchQuery(query)}
                >
                  {query}
                </Button>
              ))}
            </div>
            <div className="flex gap-2">
              <Input
                id="searchQuery"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter search query..."
              />
              <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
                {isSearching ? <LoadingSpinner /> : "Search"}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {searchResults.length > 0 && (
        <div className="space-y-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Index</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Strength</TableHead>
                <TableHead>Steered Completion</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {searchResults.map((result, index) => (
                <TableRow key={index}>
                  <TableCell>{result.modelId}</TableCell>
                  <TableCell>{result.layer}</TableCell>
                  <TableCell>{result.index}</TableCell>
                  <TableCell>{result.description}</TableCell>
                  <TableCell>{result.neuron.maxActApprox.toFixed(2)}</TableCell>
                  <TableCell>
                    {result.isLoading ? (
                      <div className="flex items-center gap-2">
                        <span>{steerPrompt}</span>
                        <LoadingSpinner />
                      </div>
                    ) : (
                      <span>{result.steeredCompletion}</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {complete && (
            <div className="flex flex-col items-center space-y-4">
              <div className="text-lg font-medium text-green-600">
                Steerify Complete
              </div>
              <Button onClick={handleReset}>Reset</Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}