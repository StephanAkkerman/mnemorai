import Image from "next/image";
import React, { useEffect, useState, useRef } from "react";
import { Speaker, Edit2, Check, X } from "lucide-react";
import { Card } from "@/interfaces/CardInterfaces";
import Button from "@/components/ui/Button";

interface FlashcardProps {
  card: Card;
  isLoading: boolean;
  showFront?: boolean;
  disableEdit?: boolean;
  onCardUpdate?: (updatedCard: Card) => Promise<void>;
  className?: string;
}

export default function Flashcard({
  card,
  isLoading,
  showFront = false,
  disableEdit = true,
  onCardUpdate,
  className
}: FlashcardProps) {
  const [flipped, setFlipped] = useState(!showFront);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioError, setAudioError] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Edit mode states
  const [isEditing, setIsEditing] = useState(false);
  const [editedWord, setEditedWord] = useState(card.word);
  const [editedVerbalCue, setEditedVerbalCue] = useState(card.verbalCue);
  const [editedIpa, setEditedIpa] = useState(card.ipa);

  // Add state for saving status
  const [isSaving, setIsSaving] = useState(false);


  useEffect(() => {
    setFlipped(!showFront);
  }, [showFront]);

  useEffect(() => {
    if (isLoading) {
      setFlipped(false);
    } else if (card.imageUrl) {
      setFlipped(false);
    }
  }, [isLoading, card.imageUrl]);

  // Update edited values when card changes
  useEffect(() => {
    setEditedWord(card.word);
    setEditedVerbalCue(card.verbalCue);
    setEditedIpa(card.ipa);
  }, [card]);

  // Set up audio event listeners
  useEffect(() => {
    const audioElement = audioRef.current;
    if (!audioElement) return;

    const handlePlay = () => setIsPlaying(true);
    const handleEnded = () => setIsPlaying(false);
    const handleError = () => {
      setAudioError(true);
      setIsPlaying(false);
      console.error("Audio playback error");
    };

    audioElement.addEventListener("play", handlePlay);
    audioElement.addEventListener("ended", handleEnded);
    audioElement.addEventListener("error", handleError);

    return () => {
      audioElement.removeEventListener("play", handlePlay);
      audioElement.removeEventListener("ended", handleEnded);
      audioElement.removeEventListener("error", handleError);
    };
  }, []);

  // Reset audio error state when card changes
  useEffect(() => {
    if (card.audioUrl && audioRef.current) {
      audioRef.current.src = card.audioUrl;
      setAudioError(false);
    }
  }, [card.audioUrl]);

  const toggleFlip = () => {
    if (!isEditing) {
      setFlipped(!flipped);
    }
  };

  // And update the playAudio function to be more robust
  const playAudio = async (e: React.MouseEvent) => {
    e.stopPropagation();

    if (!card.audioUrl) {
      console.error("No audio URL available for playback");
      setAudioError(true);
      return;
    }

    if (!audioRef.current) {
      console.error("Audio element reference not available");
      setAudioError(true);
      return;
    }

    try {
      setAudioError(false);

      // Ensure the audio source is set to the current card's audio URL
      if (audioRef.current.src !== card.audioUrl) {
        audioRef.current.src = card.audioUrl;
      }

      // The play() method returns a promise
      await audioRef.current.play();
      setIsPlaying(true);
    } catch (error) {
      console.error("Error playing audio:", error);
      setAudioError(true);
      setIsPlaying(false);
    }
  };

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(true);
  };

  const handleSave = async () => {
    if (onCardUpdate) {
      setIsSaving(true);

      // Important: Make sure we preserve the audioUrl when updating the card
      const updatedCard: Card = {
        ...card,
        word: editedWord,
        verbalCue: editedVerbalCue,
        ipa: editedIpa,
        // Explicitly preserve these fields to ensure they don't get lost
        audioUrl: card.audioUrl,
        imageUrl: card.imageUrl,
        noteId: card.noteId
      };

      try {
        await onCardUpdate(updatedCard);
        // No need to do anything else since the parent component will handle the update
      } catch (error) {
        console.error("Error updating card:", error);
        // If there was an error, we could show a message, but we'll still exit edit mode
      } finally {
        setIsSaving(false);
        setIsEditing(false);
      }
    } else {
      setIsEditing(false);
    }
  };

  const handleCancel = () => {
    // Reset to original values
    setEditedWord(card.word);
    setEditedVerbalCue(card.verbalCue);
    setEditedIpa(card.ipa);
    setIsEditing(false);
  };

  // Prevent clicking the card to flip when in edit mode
  const cardClickHandler = isEditing ? (e: React.MouseEvent) => e.stopPropagation() : toggleFlip;

  return (
    <div
      className={`relative w-80 h-96 perspective cursor-pointer group ${className}`}
      onClick={cardClickHandler}
    >
      <audio ref={audioRef} />
      <div
        className={`absolute inset-0 transform transition-transform duration-700 ease-in-out ${flipped ? "rotate-y-180" : ""}`}
        style={{ transformStyle: "preserve-3d" }}
      >
        {/* Front Side */}
        <div className="absolute inset-0 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 backface-hidden flex flex-col justify-center items-center p-3">
          {isLoading ? (
            <p className="text-blue-500 font-bold animate-pulse">Loading...</p>
          ) : (
            <>
              <div className="relative w-full overflow-hidden rounded-xl">
                <Image
                  src={card.imageUrl || "https://placehold.co/400"}
                  alt={editedWord}
                  width={400}
                  height={400}
                  className="w-full h-full object-cover"
                />
              </div>
            </>
          )}
        </div>

        {/* Back Side */}
        <div
          className="absolute inset-0 bg-gradient-to-br from-blue-100 to-teal-100 dark:bg-gradient-to-br dark:from-blue-800 dark:to-teal-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 backface-hidden transform rotate-y-180 flex flex-col p-2"
          onClick={e => isEditing && e.stopPropagation()}
        >
          <div className="relative w-full h-1/2 overflow-hidden rounded-xl mb-1">
            <Image
              src={card.imageUrl || "https://placehold.co/400"}
              alt={editedWord}
              fill
              className="object-cover"
            />
          </div>
          <div className="w-full flex flex-col justify-center items-center flex-grow overflow-auto">

            {isEditing ? (
              <div className="w-full space-y-2" onClick={e => e.stopPropagation()}>
                <input
                  type="text"
                  value={editedWord}
                  onChange={(e) => setEditedWord(e.target.value)}
                  className="w-full text-2xl font-bold p-1 rounded text-center bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600"
                  placeholder="Word"
                />

                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={editedIpa}
                    onChange={(e) => setEditedIpa(e.target.value)}
                    className="flex-1 font-mono text-lg p-1 rounded text-center bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600"
                    placeholder="IPA pronunciation"
                  />
                  {card.audioUrl && (
                    <button
                      onClick={playAudio}
                      className={`p-1 rounded-full transition-colors ${isPlaying
                        ? "bg-blue-100 dark:bg-blue-800"
                        : "hover:bg-gray-200 dark:hover:bg-gray-700"
                        } ${audioError
                          ? "text-red-500 hover:bg-red-100 dark:hover:bg-red-900"
                          : "text-gray-600 dark:text-gray-300"
                        }`}
                      aria-label="Play pronunciation"
                      disabled={isPlaying}
                    >
                      <Speaker className={`w-5 h-5 ${isPlaying ? "animate-pulse" : ""}`} />
                    </button>
                  )}
                </div>

                <textarea
                  value={editedVerbalCue}
                  onChange={(e) => setEditedVerbalCue(e.target.value)}
                  className="w-full h-16 text-lg p-1 rounded bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600"
                  placeholder="Key phrase"
                />

                <div className="flex gap-2 pt-1">
                  <Button
                    onClick={handleSave}
                    className="flex-1 py-1 px-2 flex items-center justify-center gap-1"
                    variant="primary"
                    disabled={isSaving}
                  >
                    {isSaving ? (
                      <>
                        <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-1"></div>
                        <span>Saving</span>
                      </>
                    ) : (
                      <>
                        <Check size={16} />
                        <span>Save</span>
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={handleCancel}
                    className="flex-1 py-1 px-2 flex items-center justify-center gap-1"
                    variant="secondary"
                    disabled={isSaving}
                  >
                    <X size={16} />
                    <span>Cancel</span>
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <h2 className="text-lg md:text-xl font-bold text-gray-800 dark:text-gray-200 mb-1">
                  {editedWord || "Your word"}
                </h2>

                <div className="flex items-center justify-center gap-2 mb-1">
                  <p className="font-mono text-sm md:text-base text-gray-600 dark:text-gray-300">
                    {editedIpa || "IPA pronunciation"}
                  </p>
                  {card.audioUrl && (
                    <button
                      onClick={playAudio}
                      className={`p-1 rounded-full transition-colors ${isPlaying
                        ? "bg-blue-100 dark:bg-blue-800"
                        : "hover:bg-gray-200 dark:hover:bg-gray-700"
                        } ${audioError
                          ? "text-red-500 hover:bg-red-100 dark:hover:bg-red-900"
                          : "text-gray-600 dark:text-gray-300"
                        }`}
                      aria-label="Play pronunciation"
                      disabled={isPlaying}
                    >
                      <Speaker className={`w-5 h-5 ${isPlaying ? "animate-pulse" : ""}`} />
                    </button>
                  )}
                </div>

                <p className="text-sm md:text-base italic text-gray-700 dark:text-gray-300 text-center">
                  {editedVerbalCue || "This is the key phrase"}
                </p>

                {!disableEdit && (
                  <button
                    onClick={handleEdit}
                    className="absolute top-3 right-3 p-2 bg-gray-200 dark:bg-gray-700 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                  >
                    <Edit2 className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
