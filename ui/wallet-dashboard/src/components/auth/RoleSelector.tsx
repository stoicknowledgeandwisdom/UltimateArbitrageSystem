'use client'

import { UserRole } from '@/types'

interface RoleSelectorProps {
  currentRole?: UserRole
}

export function RoleSelector({ currentRole }: RoleSelectorProps) {
  const roles = ['trader', 'quant', 'compliance_officer', 'admin']
  
  return (
    <div className="flex items-center space-x-2">
      <span className="text-sm text-gray-500 dark:text-gray-400">Role:</span>
      <select className="input text-sm py-1 px-2 w-auto">
        {roles.map(role => (
          <option key={role} value={role} selected={currentRole?.name === role}>
            {role.replace('_', ' ').toUpperCase()}
          </option>
        ))}
      </select>
    </div>
  )
}

