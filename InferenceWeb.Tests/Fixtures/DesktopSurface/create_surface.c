// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <stdio.h>
#include <string.h>

static void set_integer(CFMutableDictionaryRef properties, CFStringRef key, int value) {
    CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
    CFDictionarySetValue(properties, key, number);
    CFRelease(number);
}

int main(void) {
    CFMutableDictionaryRef properties = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    set_integer(properties, kIOSurfaceWidth, 16);
    set_integer(properties, kIOSurfaceHeight, 16);
    set_integer(properties, kIOSurfaceBytesPerElement, 4);
    set_integer(properties, kIOSurfaceBytesPerRow, 64);
    set_integer(properties, kIOSurfaceAllocSize, 1024);
    IOSurfaceRef surface = IOSurfaceCreate(properties);
    CFRelease(properties);
    if (!surface) {
        fputs("IOSurfaceCreate failed\n", stderr);
        return 1;
    }
    if (IOSurfaceGetWidth(surface) != 16 || IOSurfaceGetHeight(surface) != 16 ||
        IOSurfaceGetAllocSize(surface) < 1024 || IOSurfaceLock(surface, 0, NULL) != 0) {
        CFRelease(surface);
        return 2;
    }
    void *pixels = IOSurfaceGetBaseAddress(surface);
    if (!pixels) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        return 3;
    }
    memset(pixels, 0x7f, 1024);
    int result = IOSurfaceUnlock(surface, 0, NULL);
    CFRelease(surface);
    if (result != 0) return 4;
    puts("surface-created-and-writable:16x16");
    return 0;
}
