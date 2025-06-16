export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600 mx-auto mb-4"></div>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          Loading dashboard...
        </p>
      </div>
    </div>
  )
}

