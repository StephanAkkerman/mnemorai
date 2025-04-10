interface FormFieldProps {
  label?: string;
  value: string;
  onChange?: (val: string) => void;
  children?: React.ReactNode;
  required?: boolean;
  error?: string;
}

export default function FormField({
  label,
  value,
  onChange,
  children,
  required,
  error,
}: FormFieldProps) {
  return (
    <div className="flex flex-col space-y-2">
      {label ?
        <label className="font-medium text-gray-800 dark:text-gray-200">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        :
        <></>
      }

      {children ? (
        children
      ) : (
        <input
          className={`border rounded-lg p-2 w-full bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 ${error ? "border-red-500" : "border-gray-300 dark:border-gray-600"
            }`}

          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          required={required}
        />
      )}

      {error && (
        <p className="text-red-500 text-sm dark:text-red-400">{error}</p>
      )}
    </div>
  );
}

