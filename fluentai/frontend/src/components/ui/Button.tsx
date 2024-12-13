import React from "react";

interface ButtonProps {
  text: string;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger";
}

export default function Button({ text, onClick, variant = "primary" }: ButtonProps) {
  const baseStyles = "px-4 py-2 rounded-md font-semibold";
  const variants = {
    primary: "bg-blue-500 text-white hover:bg-blue-600",
    secondary: "bg-gray-200 text-black hover:bg-gray-300",
    danger: "bg-red-500 text-white hover:bg-red-600",
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className={`${baseStyles} ${variants[variant]}`}
    >
      {text}
    </button>
  );
}
