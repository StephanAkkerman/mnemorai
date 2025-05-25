import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { createCard } from "../app/api/createCard";
import { getSupportedLanguages } from "@/app/api/languageService";
import { Card, CreateCardRequest } from "@/interfaces/CardInterfaces";
import { ModelOptions } from "@/interfaces/ModelInterface";
import { ModelService } from "@/services/modelService";

interface CardGeneratorProps {
  onCardCreated: (card: Card) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
  onWordChange: (word: string) => void;
}

const modelService = new ModelService();

export default function CardGenerator({
  onCardCreated,
  onLoading,
  onError,
  onWordChange,
}: CardGeneratorProps) {
  const [languages, setLanguages] = useState<{ [key: string]: string }>({});
  const [modelOptions, setModelOptions] = useState<ModelOptions>({
    imageModels: [],
    llmModels: []
  });
  const [input, setInput] = useState<CreateCardRequest>({
    languageCode: "",
    word: "",
    mnemonicKeyword: "",
    keySentence: "",
    imageModel: "",
    llmModel: ""
  });
  const [errors, setErrors] = useState({
    languageCode: "",
    word: ""
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        onLoading(true);
        setLoading(true);
        const [languageResponse, modelResponse] = await Promise.all([
          getSupportedLanguages(),
          modelService.getAvailableModels()
        ]);

        setLanguages(languageResponse.languages);
        setModelOptions(modelResponse);
      } catch (error) {
        console.error("Error fetching data:", error);
        onError("Failed to load data.");
      } finally {
        onLoading(false);
        setLoading(false);
      }
    };

    fetchData();
  }, [onLoading, onError]);

  const validate = () => {
    const newErrors = {
      languageCode: input.languageCode ? "" : "Language is required.",
      word: input.word ? "" : "Word is required.",
    };
    setErrors(newErrors);
    return !newErrors.languageCode && !newErrors.word;
  };

  const handleWordChange = (word: string) => {
    setInput((prev) => ({ ...prev, word }));
    onWordChange(word);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    onLoading(true);
    setLoading(true);
    onError("");

    try {
      const newCard = await createCard(input);
      onCardCreated(newCard);
    } catch (err: any) {
      onError(err.message || "An unexpected error occurred.");
    } finally {
      onLoading(false);
      setLoading(false);
    }
  };

  return (
    <>
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300">
        <div className="p-6">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent mb-6">
            Create Your Flashcard
          </h2>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Required Fields */}
            <div className="space-y-4">
              <div className="relative">
                <FormField
                  label="Language"
                  value={input.languageCode}
                  error={errors.languageCode}
                  required
                >
                  <AutoCompleteInput
                    suggestions={Object.keys(languages)}
                    onSelect={(languageName) => {
                      const languageCode = languages[languageName as keyof typeof languages];
                      setInput((prev) => ({
                        ...prev,
                        languageCode: languageCode || "",
                      }));
                    }}
                  />
                </FormField>
              </div>

              <FormField
                label="Word"
                value={input.word}
                error={errors.word}
                required
                onChange={handleWordChange}
              />
            </div>

            {/* Advanced Options Toggle */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100 
                  transition-colors px-3 py-1.5 rounded-full bg-gray-100 dark:bg-gray-900"
              >
                {showAdvanced ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                <span>Advanced Options</span>
              </button>
            </div>

            {/* Optional Fields */}
            <div
              className={`space-y-4 transition-all duration-300 ease-in-out ${showAdvanced ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                }`}
            >
              <div className="space-y-4 pt-2">
                <FormField
                  label="Key Sentence"
                  value={input.keySentence || ""}
                  onChange={(keySentence) => setInput(prev => ({ ...prev, keySentence }))}
                />

                <FormField
                  label="Mnemonic Keyword"
                  value={input.mnemonicKeyword || ""}
                  onChange={(mnemonicKeyword) => setInput(prev => ({ ...prev, mnemonicKeyword }))}
                >
                  <input
                    type="text"
                    className={`border rounded-lg p-2 w-full bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 
                      ${input.keySentence ? 'opacity-50 cursor-not-allowed' : ''}
                      ${errors.word ? "border-red-500" : "border-gray-300 dark:border-gray-600"}`}
                    value={input.mnemonicKeyword || ""}
                    onChange={(e) => setInput(prev => ({ ...prev, mnemonicKeyword: e.target.value }))}
                    disabled={!!input.keySentence}
                    placeholder={input.keySentence ? "Disabled when key sentence is provided" : "Enter a mnemonic keyword"}
                  />
                </FormField>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="relative">
                    <FormField label="Image Model" value={input.imageModel || ""}>
                      <AutoCompleteInput
                        suggestions={modelOptions.imageModels}
                        onSelect={(model) => setInput(prev => ({ ...prev, imageModel: model }))}
                        placeholder="Select or enter image model"
                      />
                    </FormField>
                  </div>

                  <div className="relative">
                    <FormField label="LLM Model" value={input.llmModel || ""}>
                      <AutoCompleteInput
                        suggestions={modelOptions.llmModels}
                        onSelect={(model) => setInput(prev => ({ ...prev, llmModel: model }))}
                        placeholder="Select or enter LLM model"
                      />
                    </FormField>
                  </div>
                </div>
              </div>
            </div>

            <Button
              text="Create Card"
              variant="primary"
              type="submit"
              disabled={loading}
              className="w-full py-3 text-lg font-bold transform hover:scale-105 transition-transform duration-200 hover:shadow-lg mt-6"
            />
          </form>
        </div>
      </div>
    </>
  );
}
