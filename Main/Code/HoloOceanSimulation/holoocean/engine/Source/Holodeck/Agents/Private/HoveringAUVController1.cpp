// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "Holodeck.h"
#include "HoveringAUVController1.h"

AHoveringAUVController1::AHoveringAUVController1(const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("HoveringAUV Controller Initialized"));
}

AHoveringAUVController1::~AHoveringAUVController1() {}
