"use client";

import { useState } from "react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import languages from "../config/languages.json";
import { createCard } from "../app/api/createCard";
import { CreateCardInterface } from "../interfaces/CreateCardInterface";

interface CardGeneratorProps {
  onCardCreated: (card: { img: string; word: string }) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
}

export default function CardGenerator({
  onCardCreated,
  onLoading,
  onError,
}: CardGeneratorProps) {
  const languagesArray = Object.keys(languages);
  const [input, setInput] = useState<CreateCardInterface>({
    language_code: "",
    word: "",
  });
  const [errors, setErrors] = useState({ language_code: "", word: "" });

  const validate = () => {
    const newErrors = {
      language_code: input.language_code ? "" : "Language is required.",
      word: input.word ? "" : "Word is required.",
    };
    setErrors(newErrors);
    return !newErrors.language_code && !newErrors.word;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    onLoading(true);
    onError("");

    try {
      const response = await createCard(input);
      onCardCreated({ img: response.imageUrl, word: input.word });
    } catch (err: any) {
      onError(err.message || "An unexpected error occurred.");
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="w-full max-w-lg p-8 bg-white rounded shadow-md">
      <h2 className="text-2xl font-semibold mb-6">Create Your Flashcard</h2>
      <form onSubmit={handleSubmit} className="grid gap-6">
        <FormField
          label="Language"
          value={input.language_code}
          error={errors.language_code}
          required
        >
          <AutoCompleteInput
            suggestions={languagesArray}
            onSelect={(languageName) => {
              const languageCode: string = languages[languageName];
              setInput((prev) => ({ ...prev, language_code: languageCode || "" }));
            }}
          />
        </FormField>

        <FormField
          label="Word"
          value={input.word}
          error={errors.word}
          required
          onChange={(word) => setInput((prev) => ({ ...prev, word }))}
        />

        <Button text="Create Card" variant="primary" onClick={handleSubmit} />
      </form>
    </div>
  );
}

