"use client";

import Link from "next/link";
import { useState } from "react";
import { set } from "zod/v4";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import Waveform from "~/components/Waveform";

interface Prediction {
  class: string;
  confidence: number
}

interface LayerData {
  shape: number[];
  values: number[];
}

interface VisualisationData {
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  visualisation: VisualisationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file)return; 
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null)
  }

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[100%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            CNN Audio Visualizer
          </h1>
          <p className="text-md mb-8 text-stone-600">
            Upload a WAV file to see the model's predictions and feauture maps
          </p>

          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav"
                id="file-upload"
                onChange={handleFileChange}
                disabled={isLoading}
                className="absolute inset-0 w-full cursor-pointer opacity-0"
              />
              <Button
                disabled={isLoading}
                className="border-stone-300"
                variant="outline"
                size="lg"
              >
                {isLoading ? "Analysing..." : "Choose File"}
              </Button>
            </div>
          </div>
        </div>   
      </div>
    </main>
  );
}
