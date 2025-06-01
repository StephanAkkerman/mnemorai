import React from "react";

interface ButtonProps {
  text?: string;
  onClick?: () => void;
  type?: "button" | "submit" | "reset";
  variant?: "primary" | "secondary" | "danger" | "success";
  className?: string;
  disabled?: boolean;
  children?: React.ReactNode;
  link?: string;
}

export default function Button({
  text,
  onClick,
  type = "button",
  variant = "primary",
  className = "",
  disabled = false,
  link,
  children,
}: ButtonProps) {
  const baseStyles =
    "px-4 py-2 rounded-md font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2";

  const variants = {
    primary: "bg-gradient-to-r from-blue-500 to-teal-400 text-white hover:from-blue-600 hover:to-teal-500 focus:ring-blue-300",
    secondary: "bg-gray-200 text-gray-800 hover:bg-gray-300 focus:ring-gray-300",
    danger: "bg-red-500 text-white hover:bg-red-600 focus:ring-red-300",
    success: "bg-green-500 text-white hover:bg-green-600 focus:ring-green-300",
  };

  const disabledStyles = disabled
    ? "opacity-50 cursor-not-allowed pointer-events-none"
    : "";


  const handleClick = () => {
    if (disabled) { return };

    onClick?.()

    if (link) {
      window.location.assign(link)
    }
  }

  return (
    <button
      type={type}
      onClick={handleClick}
      disabled={disabled}
      className={`${baseStyles} ${variants[variant]} ${disabledStyles} ${className}`}
    >
      {children || text /* Render children if provided, otherwise text */}
    </button>
  );
}

