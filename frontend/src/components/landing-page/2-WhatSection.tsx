import React, { useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { motion, useInView } from "motion/react";
import { TextGenerateEffect } from "../ui/text-generate-effect";
import Flashcard from "../Flashcard";
import { Card } from "@/interfaces/CardInterfaces";
import LanguageDock from "@/components/ui/language-dock";
import GlobeSection from "@/components/ui/globeArcs";

import duck from "../../../public/duck.jpg";


const SMALL_SCREEN_BREAKPOINT = 640;

const AnimatedFeatureCard = ({
  children,
  className,
  index,
  isInView,
}: {
  children?: React.ReactNode;
  className?: string;
  index: number;
  isInView: boolean;
}) => {
  return (
    <motion.div
      className={cn(`relative overflow-hidden h-full bg-white dark:bg-gray-950  rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700`, className)}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{
        duration: 0.5,
        delay: 0.5 + (index * 0.1),  // Staggered animation
      }}
    >
      {children}
    </motion.div>
  );
};

const FeatureTitle = ({ children }: { children?: React.ReactNode }) => {
  return (
    <p className="p-4 sm:p-6 w-full text-left tracking-tight text-black dark:text-white text-xl md:text-2xl font-semibold">
      {children}
    </p>
  );
};

const FeatureDescription = ({ children }: { children?: React.ReactNode }) => {
  return (
    <p
      className="px-4 sm:px-6 md:text-base w-full text-left mt-2 text-neutral-600 dark:text-neutral-400">
      {children}
    </p>
  );
};

export const SkeletonOne = () => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div className="mt-4 relative flex p-4 sm:p-8 w-full items-center justify-center min-h-[250px] md:min-h-[300px]">
      <div
        className="flex items-center justify-center w-full max-w-md"
        style={{
          transformStyle: "preserve-3d",
        }}
      >
        <motion.div
          className="relative rounded-lg border border-neutral-200 dark:border-neutral-700 transition-all duration-300 overflow-hidden"
          style={{
            transformStyle: "preserve-3d",
            // Apply rotation directly here
            boxShadow: isHovered
              ? "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
              : "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
          }}
          animate={isHovered ? {
            transform: "translateZ(30px) translateY(-10px)",
          } : {
            transform: "translateZ(8px) translateY(0px)",
          }}
          transition={{
            duration: 0.5,
            ease: "easeInOut",
          }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <Image
            src="/infographic.png"
            alt="infographic"
            width={500}
            height={400}
            className="object-contain w-full h-auto"
          />
        </motion.div>
      </div>
    </div>
  );
};

const FlashcardWrapper = ({ card, showFront }: { card: Card; showFront: boolean }) => {
  const [rotation, setRotation] = useState(0);

  const imageVariants = {
    whileHover: {
      scale: 1.1,
      rotate: 0,
      zIndex: 100,
    },
    whileTap: {
      scale: 1.1,
      rotate: 0,
      zIndex: 100,
    },
  };

  useEffect(() => {
    setRotation(Math.random() * 16 - 8);
  }, []);

  return (
    <motion.div
      variants={imageVariants}
      style={{
        rotate: rotation,
      }}
      whileHover="whileHover"
      whileTap="whileTap"
      className="w-full aspect-[5/6] relative cursor-pointer"
    >
      <Flashcard
        card={card}
        isLoading={false}
        showFront={showFront}
        className="w-full h-full"
      />
    </motion.div>
  );
};

export const SkeletonTwo = () => {
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  // Create multiple card data objects for different words
  const cardData = [
    {
      word: "Pato",
      imageUrl: duck.src,
      audioUrl: "",
      ipa: "/ˈpɑtoʊ/",
      verbalCue: "Spanish word for 'duck'",
      translation: "Duck",
      languageCode: "es"
    },
    {
      word: "Gato",
      imageUrl: "https://images.unsplash.com/photo-1592194996308-7b43878e84a6",
      audioUrl: "",
      ipa: "/ˈɡɑtoʊ/",
      verbalCue: "Spanish word for 'cat'",
      translation: "Cat",
      languageCode: "es"
    },
    {
      word: "Perro",
      imageUrl: "https://images.unsplash.com/photo-1517849845537-4d257902454a",
      audioUrl: "",
      ipa: "/ˈpɛroʊ/",
      verbalCue: "Spanish word for 'dog'",
      translation: "Dog",
      languageCode: "es"
    },
    {
      word: "Casa",
      imageUrl: "https://images.unsplash.com/photo-1518780664697-55e3ad937233",
      audioUrl: "",
      ipa: "/ˈkɑsɑ/",
      verbalCue: "Spanish word for 'house'",
      translation: "House",
      languageCode: "es"
    },
    {
      word: "Libro",
      imageUrl: "https://images.unsplash.com/photo-1544947950-fa07a98d237f",
      audioUrl: "",
      ipa: "/ˈlibɾo/",
      verbalCue: "Spanish word for 'book'",
      translation: "Book",
      languageCode: "es"
    },
    {
      word: "Sol",
      imageUrl: "https://images.unsplash.com/photo-1563630381190-77c336ea545a",
      audioUrl: "",
      ipa: "/sol/",
      verbalCue: "Spanish word for 'sun'",
      translation: "Sun",
      languageCode: "es"
    },
    {
      word: "Luna",
      imageUrl: "https://images.unsplash.com/photo-1578615437406-511cafe4a5c7",
      audioUrl: "",
      ipa: "/ˈluna/",
      verbalCue: "Spanish word for 'moon'",
      translation: "Moon",
      languageCode: "es"
    },
    {
      word: "Agua",
      imageUrl: "https://images.unsplash.com/photo-1538300342682-cf57afb97285",
      audioUrl: "",
      ipa: "/ˈaɡwa/",
      verbalCue: "Spanish word for 'water'",
      translation: "Water",
      languageCode: "es"
    },
    {
      word: "Arbol",
      imageUrl: "https://images.unsplash.com/photo-1520262494112-9fe481d36ec3",
      audioUrl: "",
      ipa: "/ˈarbol/",
      verbalCue: "Spanish word for 'tree'",
      translation: "Tree",
      languageCode: "es"
    },
  ];

  const baseCards = cardData.length >= 9 ? cardData : [...cardData, ...cardData, ...cardData];
  const displayCards = baseCards.slice(0, 9);
  useEffect(() => {
    const checkSize = () => {
      setIsSmallScreen(window.innerWidth < SMALL_SCREEN_BREAKPOINT);
    };

    // Initial check
    checkSize();

    // Add resize listener
    window.addEventListener("resize", checkSize);

    // Cleanup listener on component unmount
    return () => window.removeEventListener("resize", checkSize);
  }, []); // Empty dependency array ensures this runs only once on mount and cleanup on unmount

  // Determine which cards to show based on screen size
  const cardsToShow = isSmallScreen ? displayCards.slice(0, 4) : displayCards.slice(0, 9);

  return (
    // Use responsive grid columns: 2 cols on small, 3 cols on sm screens and up
    // This layout works well for both 4 cards (2x2) and 9 cards (3x3)
    <div className="mt-4 relative grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4 p-4">
      {cardsToShow.map((card, idx) => (
        <FlashcardWrapper
          key={`flashcard-${card.languageCode}-${card.word}-${idx}`} // Make key more unique if needed
          card={card}
          // Adjust flip logic if desired, e.g., maybe always show front on mobile?
          showFront={idx % 2 === 0}
        />
      ))}
    </div>
  );
};

export const SkeletonThree = () => {
  return (
    <>
      {/* LANGUAGES SECTION */}
      <div className="mt-4 p-4 sm:p-6 flex items-center justify-center">
        <div className="flex flex-wrap justify-center gap-4 max-w-md w-full">
          <LanguageDock />
        </div>
      </div>
    </>
  );
};

export const SkeletonFour = () => {
  return (
    <div className="mt-4 flex overflow-hidden items-center justify-center relative min-h-[250px] p-4">
      <GlobeSection />
    </div>
  );
};

const WhatSection = () => {
  const sectionRef = useRef(null);
  const isInView = useInView(sectionRef, { once: true, amount: 0.1 });

  const features = [
    {
      title: "Personalized Learning Paths",
      description:
        "Our AI adapts to your learning style, focusing on areas where you need the most practice to accelerate your fluency.",
      skeleton: <SkeletonOne />,
      className:
        "col-span-1 md:col-span-3",
    },
    {
      title: "Learn quicker with Smart Flashcards",
      description:
        "Generate visual and audio flashcards instantly using our advanced AI. Master vocabulary and pronunciation faster.",
      skeleton: <SkeletonTwo />,
      className: "col-span-1 md:col-span-3",
    },
    {
      title: "Expanding Language Library",
      description:
        "Currently supporting a growing list of languages. Our team is constantly working to add more!",
      skeleton: <SkeletonThree />,
      className:
        "col-span-1 md:col-span-3",
    },
    {
      title: "Learn Anytime, Anywhere",
      description:
        "Access your personalized dashboard and flashcards globally, 24/7, on any device. Your learning journey knows no bounds.",
      skeleton: <SkeletonFour />,
      className: "col-span-1 md:col-span-3",
    }
  ];

  return (
    <div className="relative z-20 py-24 sm:py-32 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <motion.div
        className="mb-16"
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
        transition={{ duration: 0.6 }}
      >
        {/* Ensure TextGenerateEffect component exists and works */}
        <h1 className="md:leading-tight max-w-5xl mx-auto text-center tracking-tight font-medium text-black dark:text-white">
          <TextGenerateEffect words={'Learning languages has never been easier'} className="text-3xl md:text-4xl lg:text-5xl" />
        </h1>

        <motion.p
          className="text-base md:text-lg max-w-3xl mt-6 mx-auto text-neutral-500 text-center font-normal dark:text-neutral-400"
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >Leverage AI to create personalized flashcards, follow adaptive learning paths, and practice pronunciation, making language acquisition intuitive and effective.</motion.p>
      </motion.div>

      <div className="relative" ref={sectionRef} >
        <div className="grid grid-cols-1 md:grid-cols-6 gap-6 md:gap-8">
          {features.map((feature, index) => (
            <AnimatedFeatureCard
              key={feature.title}
              className={feature.className}
              index={index}
              isInView={isInView}
            >
              <FeatureTitle>{feature.title}</FeatureTitle>
              <FeatureDescription>{feature.description}</FeatureDescription>
              {feature.skeleton}
            </AnimatedFeatureCard>
          ))}
        </div>
      </div>
    </div>
  );
};

export default WhatSection;
