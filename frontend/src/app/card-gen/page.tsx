"use client";

import { useState } from "react";
import CardGenerator from "../../components/CardGenerator";
import Flashcard from "../../components/Flashcard";
import SaveToAnki from "@/components/SaveToAnki";
import { Card } from "@/interfaces/CardInterfaces";

export default function CardGeneratorPage() {
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentWord, setCurrentWord] = useState("");

  const defaultCard: Card = {
    word: currentWord || "Your word",
    translation: "This is the English word.",
    verbalCue: "This is the key phrase",
    imageUrl: "https://placehold.co/400",
    audioUrl: "",
    ipa: "jʊər wɜrd",
    languageCode: "en"
  };

  return (
    <div className="flex-grow max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-12 space-y-8">
      <div className="flex flex-col gap-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          <div className="flex gap-12 flex-col">
            <CardGenerator
              onCardCreated={setCard}
              onLoading={setLoading}
              onError={setError}
              onWordChange={setCurrentWord}
            />
            {error && <p className="text-red-500 font-medium mt-4">{error}</p>}
          </div>
          <div className="flex items-center justify-center">
            <Flashcard
              card={card || defaultCard}
              isLoading={loading}
              showFront={!!card}
              disableEdit={false}
            />
          </div>
          {card && (
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 mt-8">
              <div className="p-6">
                <h3 className="text-xl font-bold mb-4">Save to Anki</h3>
                <SaveToAnki card={card} onError={setError} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
