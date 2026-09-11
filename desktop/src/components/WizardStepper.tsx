import React from 'react';
import { Check, Lock } from 'lucide-react';

export type WizardStep = 'restore' | 'generate' | 'results';

export interface WizardStepperProps {
  currentStep: WizardStep;
  completedSteps: WizardStep[];
  onStepClick: (step: WizardStep) => void;
}

interface StepMeta {
  id: WizardStep;
  num: number;
  label: string;
  sublabel: string;
}

const STEPS: StepMeta[] = [
  { id: 'restore', num: 1, label: 'Khôi phục ảnh cũ', sublabel: 'CodeFormer & Chọn mặt' },
  { id: 'generate', num: 2, label: 'Sinh ảnh & Đối soát', sublabel: 'FADING Inversion & Search' },
  { id: 'results', num: 3, label: 'Kết quả đối soát', sublabel: 'Bằng chứng & Đánh giá' },
];

export const WizardStepper: React.FC<WizardStepperProps> = ({
  currentStep,
  completedSteps,
  onStepClick,
}) => {
  return (
    <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-4 shadow-md">
      <div className="flex items-center justify-between relative">
        {STEPS.map((step, idx) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === step.id;
          const isLocked = !isCompleted && !isCurrent;
          const canClick = isCompleted && !isCurrent;

          return (
            <React.Fragment key={step.id}>
              {idx > 0 && (
                <div
                  data-testid={`connector-${idx}`}
                  className={`flex-1 h-0.5 mx-3 transition-colors duration-300 ${
                    completedSteps.includes(STEPS[idx - 1].id)
                      ? 'bg-[#4A8FA0]'
                      : 'bg-[#262E38]'
                  }`}
                />
              )}

              <button
                type="button"
                data-testid={`wizard-step-${step.id}`}
                aria-label={`Bước ${step.num}: ${step.label}`}
                disabled={isLocked || isCurrent}
                onClick={() => {
                  if (canClick) {
                    onStepClick(step.id);
                  }
                }}
                className={`flex items-center gap-3 text-left transition-all px-2.5 py-1.5 rounded-xl ${
                  canClick
                    ? 'cursor-pointer hover:bg-[#12161C] hover:ring-1 hover:ring-[#4A8FA0]/40 group'
                    : isCurrent
                    ? 'cursor-default'
                    : 'cursor-not-allowed opacity-40 pointer-events-none'
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center font-bold text-xs transition-all border ${
                    isCurrent
                      ? 'bg-[#C97B4A] border-[#C97B4A] text-[#E8E6E0] shadow-md ring-2 ring-[#C97B4A]/30'
                      : isCompleted
                      ? 'bg-[#4A8FA0]/20 border-[#4A8FA0] text-[#4A8FA0] group-hover:scale-105 group-hover:bg-[#4A8FA0]/30'
                      : 'bg-[#12161C] border-[#262E38] text-[#8E98A5]'
                  }`}
                >
                  {isCompleted ? (
                    <Check className="w-4 h-4 text-[#4A8FA0] stroke-[2.5]" />
                  ) : isLocked ? (
                    <Lock className="w-3.5 h-3.5 text-[#8E98A5]" />
                  ) : (
                    <span>{step.num}</span>
                  )}
                </div>

                <div className="hidden sm:block">
                  <p
                    className={`text-xs font-semibold ${
                      isCurrent
                        ? 'text-[#E8E6E0]'
                        : isCompleted
                        ? 'text-[#4A8FA0]'
                        : 'text-[#8E98A5]'
                    }`}
                  >
                    {step.label}
                  </p>
                  <p className="text-[10px] text-[#8E98A5] mt-0.5">{step.sublabel}</p>
                </div>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
