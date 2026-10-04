// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Threading;

namespace TensorSharp
{
    /// <summary>
    /// Provides a thread safe reference counting implementation. Inheritors need only implement the Destroy() method,
    /// which will be called when the reference count reaches zero. The reference count automatically starts at 1.
    /// </summary>

    [Serializable]
    public abstract class RefCounted
    {
        private int refCount = 1;

        internal virtual object? ReferenceMutationGate => null;
        internal virtual void ValidateReferenceAddition() { }
        internal virtual bool IsRetainedFinalizerFailure(Exception error) => false;
        internal int ReadReferenceCount() => Volatile.Read(ref refCount);

        /// <summary>
        /// Construct a new reference counted object. The reference count automatically starts at 1.
        /// </summary>
        public RefCounted()
        {
        }

        ~RefCounted()
        {
            try
            {
                if (ReadReferenceCount() > 0)
                {
                    Destroy();
                    Volatile.Write(ref refCount, 0);
                }
            }
            catch (Exception error) when (IsRetainedFinalizerFailure(error)) { }
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected abstract void Destroy();

        /// <summary>
        /// Returns true if the object has already been destroyed; false otherwise.
        /// </summary>
        /// <returns>true if the object is destroyed; false otherwise.</returns>
        protected bool IsDestroyed()
        {
            return ReadReferenceCount() == 0;
        }

        /// <summary>
        /// Throws an exception if the object has been destroyed, otherwise does nothing.
        /// </summary>
        protected void ThrowIfDestroyed()
        {
            if (IsDestroyed())
            {
                throw new InvalidOperationException("Reference counted object has been destroyed");
            }
        }

        protected int GetCurrentRefCount()
        {
            return ReadReferenceCount();
        }

        /// <summary>
        /// Increments the reference count. If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void AddRef()
        {
            object? gate = ReferenceMutationGate;
            if (gate != null) Monitor.Enter(gate);
            try
            {
                SpinWait spin = new SpinWait();
                while (true)
                {
                    int current = ReadReferenceCount();
                    if (current == 0)
                        throw new InvalidOperationException("Cannot AddRef - object has already been destroyed");
                    if (current == int.MaxValue)
                        throw new OverflowException("Cannot AddRef - reference count exceeds Int32.MaxValue");

                    ValidateReferenceAddition();
                    if (Interlocked.CompareExchange(ref refCount, current + 1, current) == current)
                        return;
                    spin.SpinOnce();
                }
            }
            finally
            {
                if (gate != null) Monitor.Exit(gate);
            }
        }

        /// <summary>
        /// Decrements the reference count. If the reference count reaches zero, the object is destroyed.
        /// If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void Release()
        {
            object? gate = ReferenceMutationGate;
            bool destroy = false;
            if (gate != null) Monitor.Enter(gate);
            try
            {
                SpinWait spin = new SpinWait();
                while (true)
                {
                    int current = ReadReferenceCount();
                    if (current == 0)
                        throw new InvalidOperationException("Cannot release object - object has already been destroyed");
                    if (Interlocked.CompareExchange(ref refCount, current - 1, current) == current)
                    {
                        destroy = current == 1;
                        break;
                    }
                    spin.SpinOnce();
                }
            }
            finally
            {
                if (gate != null) Monitor.Exit(gate);
            }

            if (destroy) Destroy();
        }
    }
}
