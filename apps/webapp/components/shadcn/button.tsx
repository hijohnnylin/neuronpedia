/* eslint-disable react/jsx-props-no-spreading */

import { cn } from '@/lib/utils/ui';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import * as React from 'react';

const buttonVariants = cva(
  'inline-flex items-center justify-center text-slate-600 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-sky-700 text-white shadow hover:bg-sky-700/90',
        emerald: 'bg-emerald-700 text-white shadow hover:bg-emerald-700/90',
        destructive: 'bg-red-700 text-white shadow hover:bg-red-700/90',
        outline: 'border border-input bg-white hover:bg-slate-100 hover:text-slate-700',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        slateLight: 'bg-slate-200 text-slate-500 hover:bg-slate-200/90',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-sky-700 underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2',
        xs: 'rounded-sm px-2 text-[10px]',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-6',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = 'Button';

export { Button, buttonVariants };
